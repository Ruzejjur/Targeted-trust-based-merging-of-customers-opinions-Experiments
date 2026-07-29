import io
import logging
import sys
import collections
from functools import partial
from tqdm.auto import tqdm as std_tqdm
from tqdm.contrib.logging import logging_redirect_tqdm 

class TqdmToLogger(io.StringIO):
    """Redirects tqdm output to a standard Python logger."""
    def __init__(self, logger=None, level=logging.INFO):
        super().__init__()
        self.logger = logger or logging.getLogger()
        self.level = level
        # Remember the last 10 unique prefixes to support nested progress bars
        self.last_prefixes = collections.deque(maxlen=10)

    def write(self, buf: str) -> int:
        clean_buf = buf.strip('\r\n\t ')
        if clean_buf and not clean_buf.startswith('\x1b'):
            
            # Extract everything before the time bracket [...]
            prefix = clean_buf.rsplit('[', 1)[0]
            
            # If we haven't seen this exact progress state recently, log it
            if prefix not in self.last_prefixes:
                self.logger.log(self.level, clean_buf)
                self.last_prefixes.append(prefix)
        return len(buf)
            
    def flush(self):
        pass

tqdm_out = TqdmToLogger()
tqdm = partial(std_tqdm, file=tqdm_out, mininterval=30.0, maxinterval=30.0, ascii=True)