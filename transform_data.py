import sys
import py7zr
import os
from stl.mesh import Mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tqdm import tqdm
import shutil

# RESOLUTION = (64, 64)

def extract(path):
    with py7zr.SevenZipFile(path) as f:
        os.mkdir('temp')
        f.extractall('temp')

def get_2d(path, name, dirpath):
    m = Mesh.from_file(os.path.join(path, name))
    fig = plt.figure(figsize=(8, 8), dpi=8)
    ax = fig.gca(projection='shapenet')
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors, color='white'))
    scale = m.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    plt.axis('off')
    ax.view_init(azim=0)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    fig.savefig(os.path.join(dirpath, name[:-4] + '.jpg'))
    plt.close(fig)


def main(argv):
    archive_name = argv[1]
    dirpath = argv[2]
    if not os.path.exists('temp'):
        extract(archive_name)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    files = []
    for root, _, fs in os.walk('temp'):
        files.extend((root, f) for f in fs)
    for f in tqdm(files):
        get_2d(f[0], f[1], dirpath)
    shutil.rmtree('temp')


if __name__ == '__main__':
    main(sys.argv)
