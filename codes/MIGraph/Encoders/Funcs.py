def is_num(s):
    s=str(s)
    return s.replace(',', '').replace('.', '').replace('-', '').replace('E', '').replace('e', '').isnumeric()

