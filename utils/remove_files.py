import os


def remove_files_by_name(starts_with='noise'):
    files = [f for f in os.listdir('..') if
             os.path.isfile(f) and f.startswith(starts_with)]
    for f in files:
        os.remove(f)


if __name__ == "__main__":
    remove_files_by_name(starts_with='noise')
