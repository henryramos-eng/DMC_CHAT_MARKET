"""Interfaz de Telegram para el RAG conversacional y boletines PNG."""

from __future__ import annotations

import asyncio
import logging
from typing import BinaryIO

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Message, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from .config import RAGConfig
from .dates import DateQueryError, display_date, parse_query_date
from .intents import normalize
from .service import RAGReply, RAGService


LOGGER = logging.getLogger("rag_telegram")
DATE_CALLBACK_PREFIX = "ragdate:"


class TelegramRAGBot:
    def __init__(self, config: RAGConfig) -> None:
        if not config.telegram_bot_token:
            raise ValueError("Falta TELEGRAM_BOT_TOKEN")
        self.config = config
        self.service = RAGService(config)
        self._pending_image_bulletins: dict[int, tuple[str, ...]] = {}
        self.application = Application.builder().token(config.telegram_bot_token).build()
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("boletin", self.bulletin))
        self.application.add_handler(CallbackQueryHandler(
            self.date_selection, pattern=rf"^{DATE_CALLBACK_PREFIX}\d{{4}}-\d{{2}}-\d{{2}}$"))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message))
        self.application.add_error_handler(self.on_error)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if update.effective_chat and update.effective_message:
            reply = await self.service.handle_reply(update.effective_chat.id, "Hola")
            await _send_reply(update.effective_message, reply)

    async def bulletin(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        message, chat = update.effective_message, update.effective_chat
        if not message or not chat:
            return
        if not context.args:
            await self._offer_bulletin_dates(message, context, chat.id)
            return
        try:
            parsed = parse_query_date(" ".join(context.args))
        except DateQueryError as exc:
            await message.reply_text(str(exc))
            return
        if parsed.publication_date is None:
            await self._offer_bulletin_dates(message, context, chat.id)
            return
        self._pending_image_bulletins.pop(chat.id, None)
        await self._send_bulletin_images(message, context, chat.id, parsed.publication_date)

    async def message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        message, chat = update.effective_message, update.effective_chat
        if not message or not chat or not message.text:
            return

        pending_dates = self._pending_image_bulletins.get(chat.id)
        if pending_dates:
            selected = _select_pending_date(message.text, pending_dates)
            if selected:
                self._pending_image_bulletins.pop(chat.id, None)
                await self._send_bulletin_images(message, context, chat.id, selected)
                return
            self._pending_image_bulletins.pop(chat.id, None)

        if _wants_image_bulletin(message.text):
            try:
                parsed = parse_query_date(message.text)
            except DateQueryError as exc:
                await message.reply_text(str(exc))
                return
            if parsed.publication_date:
                await self._send_bulletin_images(message, context, chat.id, parsed.publication_date)
                return
            await self._offer_bulletin_dates(message, context, chat.id)
            return

        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
        reply = await self.service.handle_reply(chat.id, message.text)
        await _send_reply(message, reply)

    async def date_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        query, chat = update.callback_query, update.effective_chat
        if not query or not query.data or not chat or not query.message:
            return
        await query.answer()
        await query.edit_message_reply_markup(reply_markup=None)
        publication_date = query.data.removeprefix(DATE_CALLBACK_PREFIX)
        if chat.id in self._pending_image_bulletins:
            self._pending_image_bulletins.pop(chat.id, None)
            await self._send_bulletin_images(query.message, context, chat.id, publication_date)
            return
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
        reply = await self.service.select_date(chat.id, publication_date)
        await _send_reply(query.message, reply)

    async def _offer_bulletin_dates(self, message: Message, context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
        """Muestra botones solo para fechas válidas en ambas fuentes."""
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        try:
            from bulletin.config import BulletinConfig
            from bulletin.pipeline import BulletinPipeline

            pipeline = BulletinPipeline(BulletinConfig.from_env(), self.config)
            dates = await asyncio.to_thread(pipeline.available_dates)
            if not dates:
                await message.reply_text("No hay fechas comunes disponibles entre los reportes y el Excel técnico.")
                return
            self._pending_image_bulletins[chat_id] = dates
            await message.reply_text("Selecciona la fecha del boletín:", reply_markup=_date_keyboard(dates))
        except Exception as exc:
            LOGGER.exception("No se pudieron listar las fechas del boletín", exc_info=exc)
            detail = str(exc).strip() or exc.__class__.__name__
            await message.reply_text(f"No se pudieron consultar las fechas disponibles. Detalle: {detail[:300]}")

    async def _send_bulletin_images(self, message: Message, context: ContextTypes.DEFAULT_TYPE,
                                    chat_id: int, publication_date: str | None) -> None:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
        status = await message.reply_text("Generando las cinco imágenes del boletín...")
        try:
            from bulletin.config import BulletinConfig
            from bulletin.pipeline import BulletinPipeline

            pipeline = BulletinPipeline(BulletinConfig.from_env(), self.config)
            result = await asyncio.to_thread(pipeline.run, publication_date)
            handles: list[BinaryIO] = []
            try:
                media: list[InputMediaPhoto] = []
                for index, path in enumerate(result.images):
                    handle = path.open("rb")
                    handles.append(handle)
                    caption = f"Boletín {display_date(result.publication_date)}." if index == 0 else None
                    media.append(InputMediaPhoto(media=handle, caption=caption))
                await context.bot.send_media_group(chat_id=chat_id, media=media)
            finally:
                for handle in handles:
                    handle.close()
            await status.edit_text("Boletín generado y enviado: 5 imágenes.")
        except Exception as exc:
            LOGGER.exception("No se pudo generar el boletín", exc_info=exc)
            detail = str(exc).strip() or exc.__class__.__name__
            await status.edit_text("No se pudo generar el boletín en imágenes. " f"Detalle: {detail[:500]}")

    async def _authorized(self, update: Update) -> bool:
        user, message = update.effective_user, update.effective_message
        if not user or not message:
            return False
        allowed = self.config.allowed_telegram_users
        if allowed and user.id not in allowed:
            await message.reply_text("No tienes autorización para utilizar este bot.")
            return False
        return True

    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        LOGGER.exception("Error al procesar una actualización", exc_info=context.error)

    def run(self) -> None:
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)


def _wants_image_bulletin(text: str) -> bool:
    value = normalize(text)
    return "boletin" in value or ("imagen" in value and any(term in value for term in ("resumen", "mercado", "tecnico")))


def _select_pending_date(text: str, dates: tuple[str, ...]) -> str | None:
    value = text.strip()
    if value.isdigit() and len(value) <= 2:
        option = int(value)
        return dates[option - 1] if 1 <= option <= len(dates) else None
    try:
        parsed = parse_query_date(value)
    except DateQueryError:
        return None
    return parsed.publication_date if parsed.publication_date in dates else None


async def _send_reply(message: Message, reply: RAGReply) -> None:
    parts = _split_telegram_message(reply.text)
    markup = _date_keyboard(reply.date_options) if reply.date_options else None
    for index, part in enumerate(parts):
        await message.reply_text(part, reply_markup=markup if index == len(parts) - 1 else None)


def _date_keyboard(dates: tuple[str, ...]) -> InlineKeyboardMarkup:
    buttons = [InlineKeyboardButton(display_date(value), callback_data=f"{DATE_CALLBACK_PREFIX}{value}") for value in dates]
    return InlineKeyboardMarkup([buttons[index:index + 2] for index in range(0, len(buttons), 2)])


def _split_telegram_message(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, remaining = [], text
    while remaining:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return parts


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    TelegramRAGBot(RAGConfig.from_env()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
