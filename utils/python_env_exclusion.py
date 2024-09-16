gitignore = open(".gitignore", "r")
if "graphEnv" not in gitignore.read():
    gitignore.close()
    gitignore = open(".gitignore", "a")
    gitignore.write("\ngraphEnv\n")
    gitignore.close()