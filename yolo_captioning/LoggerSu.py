
import sys
 
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")
 
    def write(self, message):
        self.terminal.write(message+'\n')
        self.terminal.flush()
        self.log.write(message+'\n')
        self.log.flush()
 
    def flush(self):
        pass
