import re
import sys

def main():
    filename = sys.argv[1]
    code = " "
    with open(filename) as f:
        preprocessor = False
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split("//")[0].strip()
            if not line:
                continue

            if line[0] == "#":
                if code[-1] != "\n":
                    code += "\n"
                if line[-1] == "\\":
                    code += line[:-1]
                    code += " "
                    preprocessor = True
                else:
                    code += line
                    code += "\n"
            else:
                if preprocessor:
                    if line[-1] == "\\":
                        code += line[:-1]
                        code += " "
                    else:
                        code += line
                        code += "\n"
                        preprocessor = False
                else:
                    code += line
                    code += " "
    
    code = code.strip()

    # コメントの除去
    code = re.sub(r"/\*.*?\*/", " ", code)

    # 連続した改行以外のスペースの除去
    code = re.sub(r"[^\S\n\r]+", " ", code)

    # 改行に隣接するスペースの除去
    code = re.sub(r"\s?\n\s?", "\n", code)

    print(code)

if __name__ == "__main__":
    main()
