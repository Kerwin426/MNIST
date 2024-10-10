from cx_Freeze import setup, Executable

# 添加你的脚本名称和其他配置
executables = [Executable("./gui.py")]

setup(
    name="MNIST_RE",
    version="0.1",
    description="描述你的应用",
    executables=executables
)
