"""Interfaz de Telegram para el RAG conversacional."""

from __future__ import annotations

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import RAGConfig
from .dates import display_date
from .service import RAGReply, RAGService


LOGGER = logging.getLogger("rag_telegram")
DATE_CALLBACK_PREFIX = "ragdate:"


class TelegramRAGBot:
    def __init__(self, config: RAGConfig) -> None:
        if not config.telegram_bot_token:
            raise ValueError("Falta TELEGRAM_BOT_TOKEN")
        self.config = config
        self.service = RAGService(config)
        self.application = Application.builder().token(config.telegram_bot_token).build()
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(
            CallbackQueryHandler(
                self.date_selection,
                pattern=rf"^{DATE_CALLBACK_PREFIX}\d{{4}}-\d{{2}}-\d{{2}}$",
            )
        )
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.message)
        )
        self.application.add_error_handler(self.on_error)

    async def start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._authorized(update):
            return
        if update.effective_chat and update.effective_message:
            reply = await self.service.handle_reply(update.effective_chat.id, "Hola")
            await _send_reply(update.effective_message, reply)

    async def message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._authorized(update):
            return
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat or not message.text:
            return
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
        reply = await self.service.handle_reply(chat.id, message.text)
        await _send_reply(message, reply)

    async def date_selection(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._authorized(update):
            return
        query = update.callback_query
        chat = update.effective_chat
        if not query or not query.data or not chat or not query.message:
            return

        await query.answer()
        await query.edit_message_reply_markup(reply_markup=None)
        publication_date = query.data.removeprefix(DATE_CALLBACK_PREFIX)
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
        reply = await self.service.select_date(chat.id, publication_date)
        await _send_reply(query.message, reply)

    async def _authorized(self, update: Update) -> bool:
        user = update.effective_user
        message = update.effective_message
        if not user or not message:
            return False
        allowed = self.config.allowed_telegram_users
        if allowed and user.id not in allowed:
            await message.reply_text("No tienes autorización para utilizar este bot.")
            return False
        return True

    async def on_error(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        LOGGER.exception("Error al procesar una actualización", exc_info=context.error)

    def run(self) -> None:
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)


async def _send_reply(message: Message, reply: RAGReply) -> None:
    parts = _split_telegram_message(reply.text)
    markup = _date_keyboard(reply.date_options) if reply.date_options else None
    for index, part in enumerate(parts):
        is_last = index == len(parts) - 1
        await message.reply_text(
            part,
            reply_markup=markup if is_last else None,
        )


def _date_keyboard(dates: tuple[str, ...]) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            display_date(value),
            callback_data=f"{DATE_CALLBACK_PREFIX}{value}",
        )
        for value in dates
    ]
    rows = [buttons[index : index + 2] for index in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(rows)


def _split_telegram_message(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    remaining = text
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
