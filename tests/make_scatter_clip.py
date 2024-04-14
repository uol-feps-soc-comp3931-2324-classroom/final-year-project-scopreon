import os
import time

from simple_worm.plot3d import generate_scatter_clip
from simple_worm.plot2d import *
from tests.helpers import generate_test_target


def test_scatter_clip():
    # Make a single video clip
    MP, F0, CS, FS = generate_test_target(
        N=48,
        T=3,
        dt=0.001
    )
    save_dir = 'vids'
    save_fn = time.strftime('%Y-%m-%d_%H%M%S')
    generate_scatter_clip(
        clips=[FS[0].to_numpy()],
        save_dir=save_dir,
        save_fn=save_fn
    )

    FS_to_midline_csv(FS[0])
    print(len(FS))
    clip_midline_csv()
    # plot_midline(FS[0])

    # Check the video file was created
    assert os.path.exists(save_dir + '/' + save_fn + '.mp4')


if __name__ == '__main__':
    test_scatter_clip()
