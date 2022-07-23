import csv


with open("../patches_and_shit24622.csv","w") as patch_csv:
    writer = csv.writer(patch_csv)
    with open("../patches/image_csv.csv","r") as srgb_csv:
        srgb_reader = csv.reader(srgb_csv)
        for i,path in enumerate(srgb_reader):
            if not i%3:
                writer.writerow(path)

    with open("../refined_synt/syn_csv.csv", "r") as syn_csv:
        raw_reader = csv.reader(syn_csv)
        for i, path in enumerate(raw_reader):
                # path[0] = path[0].replace("../../","../")
            writer.writerow(path)

