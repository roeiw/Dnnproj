import pandas as pd
# import openpyxl
import csv


def convert_2_csv():
    read_file = pd.read_excel (r'../Book2.xlsx')
    read_file.to_csv (r'../gammas_and_shit.csv', index = None, header=True)

def get_cam_iso_dict():
    iso_cam_dict={}
    with open(r'../../gammas_and_shit.csv') as csv_file:
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
        new_header = ["camera","iso","lambda_read","lambda_shot","best_mean"]
        writer.writerow(new_header)
        print(iso_cam_dict.__sizeof__())
        for key in iso_cam_dict.keys():
            cals = iso_cam_dict[key]
            camiso = key.split("_")
            camiso = camiso + cals
            writer.writerow(camiso)




