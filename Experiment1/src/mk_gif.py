from PIL import Image

frames = []

# 保存的帧数
num_frames = 200

for i in range(num_frames):
    frame = Image.open(f"D:/NJU_undergraduate/大三下/无人机/homework/GIF/parabola/parabola_{i}.png")
    frames.append(frame)

frames[0].save("D:/NJU_undergraduate/大三下/无人机/homework/GIF/parabola/parabola.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
