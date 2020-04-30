import decomposer

def run(img_path, lights_limit=1, modes=[]):

    dcmp = decomposer.Decomposer(img_path)
    dcmp.preprocess()
    dcmp.decompose(lights_limit, modes)
    dcmp.export()

    return dcmp
