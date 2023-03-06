import os, shutil


class Logging():
    """
    日志类
    """
    def __init__(self, filename):
        self.filename = filename

    def record(self, str_log):
        """
        记录日志到文件中并打印
        :param str_log:
        :return: None
        """
        filename = self.filename
        print(str_log)
        with open(filename, 'a') as f:
            f.write("%s\r\n" % str_log)
            f.flush()
