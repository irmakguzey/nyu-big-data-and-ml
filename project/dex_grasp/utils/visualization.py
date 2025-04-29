import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_hand_pose_3d(hand_pose):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    # Plot each knuckle point
    ax.scatter(hand_pose[:, 0], hand_pose[:, 1], hand_pose[:, 2], c="b", marker="o")

    # Connect points to form hand structure
    finger_colors = ["r", "g", "b", "y", "m"]
    ax.plot(
        hand_pose[0, 0],
        hand_pose[0, 1],
        hand_pose[0, 2],
        color="r",
        label="Wrist",
    )
    for i in range(5):
        start_idx = i * 4 + 1
        end_idx = start_idx + 4
        ax.plot(
            hand_pose[start_idx:end_idx, 0],
            hand_pose[start_idx:end_idx, 1],
            hand_pose[start_idx:end_idx, 2],
            color=finger_colors[i],
            linestyle=f"-",
            label=f"Finger {i+1}",
        )
        ax.plot(
            [hand_pose[0, 0], hand_pose[start_idx, 0]],
            [hand_pose[0, 1], hand_pose[start_idx, 1]],
            [hand_pose[0, 2], hand_pose[start_idx, 2]],
            color=finger_colors[i],
            linestyle="-",
            label=f"Finger {i+1}",
        )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Hand Pose Visualization (3D)")
    ax.legend()

    return fig, ax


def pt_is_oob(pt, image_size):
    return not ((0 <= pt[0] < image_size[1]) and (0 <= pt[1] < image_size[0]))


def draw_axis(
    img, pose, intrinsics, length, colors=[(0, 0, 255), (0, 255, 0), (255, 0, 0)]
):
    """Function to manually draw the axes on the detected markers"""
    img = cv2.resize(img, (intrinsics.W, intrinsics.H))
    r = pose[:3, :3]
    t = pose[:3, 3]
    r, _ = cv2.Rodrigues(r)

    # Project points from the object frame to the image frame
    axis = np.float32(
        [[length, 0, 0], [0, length, 0], [0, 0, length], [0, 0, 0]]
    ).reshape(-1, 3)
    img_pts, _ = cv2.projectPoints(axis, r, t, intrinsics.K, intrinsics.D)

    # Convert to tuples for the line function
    try:
        img_pts = [pt.ravel() for pt in img_pts]
        img_shape = img.shape[:2]
        if not pt_is_oob(img_pts[3], img_shape):
            if not pt_is_oob(img_pts[0], img_shape):
                img = cv2.line(
                    img,
                    img_pts[3].astype(np.uint32),
                    img_pts[0].astype(np.uint32),
                    colors[0],
                    3,
                )
            if not pt_is_oob(img_pts[1], img_shape):
                img = cv2.line(
                    img,
                    img_pts[3].astype(np.uint32),
                    img_pts[1].astype(np.uint32),
                    colors[1],
                    3,
                )
            if not pt_is_oob(img_pts[2], img_shape):
                img = cv2.line(
                    img,
                    img_pts[3].astype(np.uint32),
                    img_pts[2].astype(np.uint32),
                    colors[2],
                    3,
                )
    except Exception as e:
        print("having an issue with projecting points! skipping annotation")

    return img
