import os

def fix_circular_imports(directory):
    for file in os.listdir(directory):
        if file.endswith(".py"):
            path = os.path.join(directory, file)
            with open(path, "r") as f:
                content = f.read()
            
            content = content.replace("import ddar", "import ddar")

            with open(path, "w") as f:
                f.write(content)

if __name__ == "__main__":
    fix_circular_imports(".")
    print("Circular import issue fixed. Try running the scripts again.")

