import enum


class MailType(enum.Enum):
    """
    mail-types for slurm
    https://slurm.schedmd.com/sbatch.html#OPT_mail-type
    """
    NONE = enum.auto(),
    BEGIN = enum.auto(),
    END = enum.auto(),
    FAIL = enum.auto(),
    REQUEUE = enum.auto(),
    ALL = enum.auto(),
    INVALID_DEPEND = enum.auto()
    STAGE_OUT = enum.auto(),
    IME_LIMIT = enum.auto(),
    TIME_LIMIT_90 = enum.auto(),
    TIME_LIMIT_80 = enum.auto(),
    TIME_LIMIT_50 = enum.auto(),
