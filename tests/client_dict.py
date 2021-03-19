import numpy as np


def main():
    ports = [7, 2, 5]
    d = {port: False for port in ports}
    print('d: ', d)

    for port in d.keys():
        print('d (sum): ', sum(d.values()))
        if d[port] is False:
            d[port] = True

    print('d (updated): ', d)


if __name__ == "__main__":
    main()
