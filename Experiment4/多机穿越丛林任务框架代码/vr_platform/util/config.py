import configparser
import sys

config = configparser.ConfigParser()
config.read(sys.argv[1])

# from ChatGpt3.5

# 你的代码尝试读取一个配置文件，这个文件名由命令行参数给出。
# 但是，在运行代码时，你需要确保提供了正确的命令行参数，指定了要读取的配置文件的路径。
# 例如，如果你的配置文件名为config.ini，你可以在命令行中这样运行代码：
# python .\trees_go_evaluate.py config.ini
# 这样，config.ini就会作为sys.argv[1]传递给你的代码，从而正确读取配置文件。

# 训练和评价都需要加上命令行参数，即就是trees_go.ini





