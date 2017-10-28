# coding=utf8
import codecs
import glob
import os
import re
import re
import json

record_dir = 'D:/data/korean_chess/records'
output_path = 'records.txt'

position_type_map = {u'마상상마': 'masangsangma', u'마상마상': 'masangmasang', u'상마마상': 'sangmamasang',
                     u'상마상마': 'sangmasangma'}

red_position_type_map = {u'마상상마': 'masangsangma', u'마상마상': 'sangmasangma', u'상마마상': 'sangmamasang',
                         u'상마상마': 'masangmasang'}


def parse_record(tmp_records):
    tmp_record_list = tmp_records.split(' ')
    record_list = []
    for tmp_record in tmp_record_list:
        if len(tmp_record) < 5 and tmp_record != u'한수쉼':
            continue
        record_list.append(tmp_record)

    result = []
    for i, record in enumerate(record_list):
        # if record == u'한수쉼':
        #     result.append('pass')
        #     continue
        position_list = re.findall('[0-9]{2}', record)
        # print(list)
        y = int(position_list[0][0]) - 1 if int(position_list[0][0]) - 1 >= 0 else 9
        x = int(position_list[0][1]) - 1
        to_y = int(position_list[1][0]) - 1 if int(position_list[1][0]) - 1 >= 0 else 9
        to_x = int(position_list[1][1]) - 1
        if i % 2 == 1:
            # red turn! reverse xy position
            y = 9 - y
            x = 8 - x
            to_y = 9 - to_y
            to_x = 8 - to_x
        # print({'from_x': x, 'from_y': y, 'to_x': to_x, 'to_y': to_y})
        result.append({'from_x': x, 'from_y': y, 'to_x': to_x, 'to_y': to_y})
    return result


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
    i_line = 0
    for line in lines:
        i_line += 1
        line = line.strip()
        if not line:
            if tmp_records:
                record_list.append({'blue_position_type': blue_position_type,
                                    'red_position_type': red_position_type,
                                    'winner': result,
                                    'records': parse_record(tmp_records),
                                    'file': record_file_path,
                                    'line': i_line})
                tmp_records = ''
                blue_position_type = None
                red_position_type = None
                result = None
            continue

        if line[0].isdigit():
            if blue_position_type:
                if re.findall(u'한수쉼', line):
                    tmp_records = ''
                    blue_position_type = None
                    red_position_type = None
                    result = None
                else:
                    tmp_records += (' ' + line)

            continue

        if line[1:4] == u'초차림':
            print(record_file_path)
            blue_position_type = position_type_map[line[6:10]]
            continue
        if line[1:4] == u'한차림':
            red_position_type = red_position_type_map[line[6:10]]
            continue
        if line[1:4] == u'대국결':
            if re.findall('접속 끊김|시간승|무승부', line, re.UNICODE):
                blue_position_type = None
                red_position_type = None
                result = None
            else:
                result_list = re.findall('한|초', line, re.UNICODE)
                if not result_list:
                    blue_position_type = None
                    red_position_type = None
                    result = None
                else:
                    result = re.findall('한|초', line, re.UNICODE)[0]
                    result = 'b' if result == u'한' else 'r'
            continue


            # for records in record_list:
            #     print(records)

with open(output_path, 'w') as outfile:
    for records in record_list:
        outfile.write(json.dumps(records) + "\n")

print(len(record_list))
