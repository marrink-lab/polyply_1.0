import logging
from vermouth.log_helpers import (StyleAdapter, BipolarFormatter,
                                  CountingHandler, TypeAdapter,)

# Implement Logger
LOGGER = TypeAdapter(logging.getLogger('polyply'))
PRETTY_FORMATTER = logging.Formatter(fmt='{levelname:} - {type} - {message}',
                                     style='{')
DETAILED_FORMATTER = logging.Formatter(fmt='{levelname:} - {type} - {name} - {message}',
                                       style='{')
COUNTER = CountingHandler()

# Control above what level message we want to count
COUNTER.setLevel(logging.WARNING)

CONSOLE_HANDLER = logging.StreamHandler()
FORMATTER = BipolarFormatter(DETAILED_FORMATTER,
                             PRETTY_FORMATTER,
                             logging.DEBUG,
                             logger=LOGGER)

CONSOLE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(CONSOLE_HANDLER)
LOGGER.addHandler(COUNTER)

LOGGER = StyleAdapter(LOGGER)

LOGLEVELS = {0: logging.INFO, 1: logging.DEBUG, 2: 5}
