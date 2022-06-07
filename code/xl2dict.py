import pandas as pd
# import openpyxl
import csv
import statistics

def convert_2_csv():
    read_file = pd.read_excel (r'../Book2.xlsx')
    read_file.to_csv (r'../gammas_and_shit.csv', index = None, header=True)

def get_cam_iso_dict():
    iso_cam_dict={}
    with open(r'../../gammas_and_shit1.csv') as csv_file:
        reader = csv.reader(csv_file)
        header = reader.__next__()
        print(header)


        for row in reader:
            key = row[4] +"_"+ row[5]
            value = [row[1],row[3],row[6]]
            if row[0] != 'read' : continue
            if (key in iso_cam_dict.keys()):
                if iso_cam_dict[key][2]>row[6]:
                    iso_cam_dict[key] = value

            else:  iso_cam_dict[key] = value
    return iso_cam_dict

def write_csv(iso_cam_dict):
    with open('../best_lambdas.csv',"w") as csv_file:
        writer = csv.writer(csv_file)
        new_header = ["shot","read","cam","iso","best_mean","variance"]
        writer.writerow(new_header)
        print(iso_cam_dict.__sizeof__())
        for key in iso_cam_dict.keys():
            cals = iso_cam_dict[key]
            camiso = key.split("_").remove("lambda").remove("read").remove("shot")
            camiso = camiso + cals
            writer.writerow(camiso)

def create_lambda_csv(read_dict):
    with open('../lambda_and_shit.csv',"w") as csv_file:
        writer = csv.writer(csv_file)
        new_header = ["shot","read","cam","iso","mean","variance"]
        writer.writerow(new_header)
        # print(iso_cam_dict.__sizeof__())
        # for key in iso_cam_dict.keys():
        #     cals = iso_cam_dict[key]
        for key, value in read_dict.items():
            mean = statistics.mean(value)
            variance = statistics.variance(value)
            # print(key)
            # print(value)
            camiso = key.split("_")
            camiso.pop(0)
            camiso.pop(0)
            camiso.pop(1)
            camiso.pop(1)
            # print(camiso)
            # camiso = key.split("_").remove("lambda")
            # print(camiso)
            # camiso = camiso.remove("read")
            # print(camiso)

            # camiso = camiso.remove("shot")
            # print(camiso)

            camiso = camiso + [mean, variance]
            writer.writerow(camiso)



