import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


def plot_hand_pose_3d(hand_pose, linestyle="-", fig=None, ax=None):
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    if ax is None:
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
            linestyle=linestyle,
            label=f"Finger {i+1}",
        )
        ax.plot(
            [hand_pose[0, 0], hand_pose[start_idx, 0]],
            [hand_pose[0, 1], hand_pose[start_idx, 1]],
            [hand_pose[0, 2], hand_pose[start_idx, 2]],
            color=finger_colors[i],
            linestyle=linestyle,
        )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Hand Pose Visualization (3D)")
    ax.legend()

    return fig, ax


def plot_rotation(fig=None, ax=None, rotation=None, origin=np.zeros(3), linestyle="-"):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection="3d")
    # I think the rotation is the rotvec of the gt_pose (first 3 elements)
    rot_mat = R.from_rotvec(rotation[:3]).as_matrix()  # H_H_W

    # Plot arrow
    axis_scale = 0.05
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    # Rotated basis vectors
    x_rot = (rot_mat @ x_axis) * axis_scale
    y_rot = (rot_mat @ y_axis) * axis_scale
    z_rot = (rot_mat @ z_axis) * axis_scale

    # Plot rotated coordinate axes
    ax.quiver(*origin, *x_rot, color="r", linestyle=linestyle, label="x'")
    ax.quiver(*origin, *y_rot, color="g", linestyle=linestyle, label="y'")
    ax.quiver(*origin, *z_rot, color="b", linestyle=linestyle, label="z'")

    # Set limits and labels
    ax.set_xlim(-0.05, 0.15)
    ax.set_ylim(-0.10, 0.15)
    ax.set_zlim(-0.10, 0.05)
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


def draw_text(image, text, position=None, pose=None, intrinsics=None):
    # Define text and properties
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    font_scale = 0.4  # Smaller font scale
    text_color = (0, 255, 0)  # White text
    bg_color = (0, 0, 0)  # Black background
    thickness = 1  # Thin text

    # Get text size to determine background size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    if position is None:
        r = pose[:3, :3]
        t = pose[:3, 3]
        r, _ = cv2.Rodrigues(r)

        # Project points from the object frame to the image frame
        axis = np.float32([[0, 0, 0]]).reshape(-1, 3)
        img_pts, _ = cv2.projectPoints(axis, r, t, intrinsics.K, intrinsics.D)
        position = (int(img_pts[0][0][0]), int(img_pts[0][0][1]))

    bg_x1, bg_y1 = position[0] - 5, position[1] - text_height - 5  # Top-left corner
    bg_x2, bg_y2 = (
        position[0] + text_width + 5,
        position[1] + baseline + 5,
    )  # Bottom-right corner

    # Draw black rectangle as background
    image = cv2.rectangle(
        image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, thickness=cv2.FILLED
    )

    # Put text on the image
    image = cv2.putText(image, text, position, font, font_scale, text_color, thickness)

    return image
