# coding=utf8
import codecs
import glob
import os
import re


def parse_record(tmp_records):
    tmp_record_list = tmp_records.split(' ')
    record_list = []
    for tmp_record in tmp_record_list:
        if len(tmp_record) < 5:
            continue
        record_list.append(tmp_record)

    for i, record in enumerate(record_list):
        if i % 2 == 0:
            # blue turn
            y = int(record[0])
            x = int(record[1])
            to_y = int(record[-2])
            to_x = int(record[-1])
            print(1)
        else:
            # reverse xy position
            # red turn
            print(2)

    print(record_list)
    print(len(record_list))
    sys.exit()
    return tmp_records


record_dir = '/Users/softlemon/Downloads/gibo'
record_files = glob.glob(os.path.join(record_dir, '*'))

record_list = []
for record_file_path in record_files:
    record_file = codecs.open(record_file_path, encoding='euckr')

    tmp_records = ''
    blue_position_type = None
    red_position_type = None
    result = None
    try:
        lines = record_file.readlines()
    except UnicodeDecodeError:
        continue
    for line in lines:
        line = line.strip()
        if not line:
            if tmp_records:
                record_list.append({'blue_postion_type': blue_position_type,
                                    'red_position_type': red_position_type,
                                    'result': result,
                                    'records': parse_record(tmp_records)})
                tmp_records = ''
                blue_position_type = None
                red_position_type = None
            continue

        if line[0].isdigit():
            if blue_position_type:
                tmp_records += (' ' + line)
            continue

        if line[1:3] == u'초차':
            blue_position_type = line[6:10]
            continue
        if line[1:3] == u'한차':
            red_position_type = line[6:10]
            continue
        if line[1:4] == u'대국결':
            result = line
            continue


            # for records in record_list:
            #     print(records)

print len(record_list)
