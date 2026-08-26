# -*- coding: utf-8 -*-
#
# Copyright © 2024-2026 Datagniel (IfaDW)
# Licensed under the terms of the MIT License

"""Status bar widget for Ollama completion provider."""

# Standard library imports
import logging
import os

# Third party imports
from qtpy.QtCore import QPoint

# Spyder imports
from spyder.api.translations import _
from spyder.api.widgets.status import StatusBarWidget
from spyder.utils.icon_manager import ima
from spyder.api.widgets.menus import SpyderMenu
from spyder.utils.qthelpers import add_actions, create_action

# Local imports
from .config_dialog import OllamaConfigDialog

logger = logging.getLogger(__name__)


class OllamaStatusWidget(StatusBarWidget):
    """Status bar widget for Ollama completion provider."""

    BASE_TOOLTIP = _("Ollama completion status")
    DEFAULT_STATUS = _("not connected")
    ID = "ollama_status"

    def __init__(self, parent, provider):
        self.provider = provider
        self.tooltip = self.BASE_TOOLTIP
        super().__init__(parent)
        self.setVisible(True)
        self.menu = SpyderMenu(self)
        self.sig_clicked.connect(self.show_menu)

    def set_value(self, value):
        """Update displayed status."""
        ollama_enabled = self.provider.get_conf(
            ("enabled_providers", "ollama"),
            default=True,
            section="completions",
        )

        if value is not None and isinstance(value, dict) and "short" in value:
            self.tooltip = value["long"]
            value = value["short"]
        elif value is not None:
            self.setVisible(True)
        elif value is None:
            value = self.DEFAULT_STATUS
            self.tooltip = self.BASE_TOOLTIP

        self.update_tooltip()
        self.setVisible(ollama_enabled)
        value = "Ollama: {0}".format(value)
        super(OllamaStatusWidget, self).set_value(value)

    def get_tooltip(self):
        """Reimplementation to get a dynamic tooltip."""
        return self.tooltip

    def open_provider_preferences(self):
        config_dialog = OllamaConfigDialog(self.provider, parent=self)
        config_dialog.show()

    def show_menu(self):
        """Display a menu when clicking on the widget."""
        menu = self.menu
        menu.clear()
        text = _("Configure Ollama provider")
        change_action = create_action(
            self,
            text=text,
            triggered=self.open_provider_preferences,
        )
        add_actions(menu, [change_action])
        rect = self.contentsRect()
        os_height = 7 if os.name == "nt" else 12
        pos = self.mapToGlobal(
            rect.topLeft() + QPoint(-40, -rect.height() - os_height)
        )
        menu.popup(pos)

    def get_icon(self):
        """Load our icon; fall back to a stock icon if unavailable."""
        try:
            icon = ima.icon("ollama")
            if icon is not None and not icon.isNull():
                return icon
        except Exception:
            pass
        return ima.icon("help")
