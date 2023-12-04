import enum

class LogLevel(enum.Enum):
    """!The severity of a logged message.
        The log level used by error loggers. A system-wide log level controls what messages are logged.
    """
    NON = 0
    ERR = 11
    WRN = 10
    INF = 12
    DBG = 13
    TRC = 14

    def __lt__(self, other):
        # A higher value means more detailed logs
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        # A higher value means more detailed logs
        if self.__class__ is other.__class__:
            return self.value == other.value
        return NotImplemented
    
    def __le__(self, other):
        # A higher value means more detailed logs
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

# All messages at or below this log level are reported.
LOG_LEVEL = LogLevel.TRC

def report(level: LogLevel, *str):
    """!Log a message
        Report the log message if the specified log level is below the global log level. Otherwise, do nothing.
    """
    COL = ""
    COL_DEFAULT = "\033[0m"
    global LOG_LEVEL
    match level:
        case LogLevel.ERR:
            COL = "\033[31mERR: "
        case LogLevel.WRN:
            COL = "\033[33mWRN: "
        case LogLevel.INF: # Teh normal log level
            COL = "\033[34mINF: "
        case LogLevel.DBG:
            COL = "\033[36mDBG: "
        case LogLevel.TRC:
            COL = "\033[0mTRC: "
        case _:
            COL = "\035[0m[cannot interpret log level:] "
            pass

    if LOG_LEVEL <= level:
        print (*(COL, *str, COL_DEFAULT))