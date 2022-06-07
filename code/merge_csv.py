import csv


with open("../validation_and_shit.csv","w") as patch_csv:
    writer = csv.writer(patch_csv)
    with open("../patches/image_csv.csv","r") as srgb_csv:
        srgb_reader = csv.reader(srgb_csv)
        for i,path in enumerate(srgb_reader):
           if not i % 100:
                writer.writerow(path)
    with open("../syn_patch/syn_csv.csv", "r") as syn_csv:
        raw_reader = csv.reader(syn_csv)
        for i, path in enumerate(raw_reader):
            if not i % 100:
                # path[0] = path[0].replace("../../","../")
                writer.writerow(path)

