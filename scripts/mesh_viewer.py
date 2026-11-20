import sys
import os
import numpy as np
import pyvista as pv
import warnings

# Giữ log tiếng Việt không lỗi font
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    if len(sys.argv) < 2:
        print("Usage: python mesh_viewer.py <path_to_mesh.ply>")
        sys.exit(1)
        
    mesh_path = sys.argv[1]
    if not os.path.exists(mesh_path):
        print(f"Lỗi: Không tìm thấy file {mesh_path}")
        sys.exit(1)
        
    print(f"Đang mở file Mesh 3D bằng PyVista Cleaner: {mesh_path}")
    
    try:
        # Tải mesh
        original_mesh = pv.read(mesh_path)
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        sys.exit(1)
        
    # Tạo Plotter
    p = pv.Plotter(title="Mesh Cleaner - Tẩy Rác 3D")
    
    # Quản lý trạng thái (History cho Undo)
    class State:
        def __init__(self):
            self.history = []
            self.current_mesh = original_mesh.copy()
            self.actor = None
            
        def push_history(self):
            # Lưu tối đa 10 state
            self.history.append(self.current_mesh.copy())
            if len(self.history) > 10:
                self.history.pop(0)
                
        def update_mesh(self, new_mesh):
            self.push_history()
            self.current_mesh = new_mesh
            self.redraw()
            
        def undo(self):
            if len(self.history) > 0:
                self.current_mesh = self.history.pop()
                self.redraw()
                print("Đã hoàn tác (Undo).")
            else:
                print("Không có lịch sử để hoàn tác.")
                
        def redraw(self):
            if self.actor is not None:
                p.remove_actor(self.actor)
            # Show edges helps user see faces
            self.actor = p.add_mesh(self.current_mesh, show_edges=False, smooth_shading=True)

    state = State()
    state.redraw()
    
    # Nút Tẩy Tự Động (Auto Clean)
    def btn_auto_clean(is_checked):
        print("Đang dọn dẹp tự động...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Tự động chọn cụm tam giác lớn nhất (bỏ qua rác vụn)
            clean_mesh = state.current_mesh.connectivity('largest').extract_surface()
        state.update_mesh(clean_mesh)
        print("Tẩy Tự Động: Đã giữ lại khối Mesh lớn nhất!")
        
    # Nút Hoàn Tác (Undo)
    def btn_undo(is_checked):
        state.undo()

    # Nút Lưu (Save)
    def btn_save(is_checked):
        print(f"Đang lưu đè file: {mesh_path}")
        state.current_mesh.save(mesh_path)
        print("Lưu thành công!")
        p.close()

    # Thêm các nút lên màn hình
    p.add_checkbox_button_widget(btn_auto_clean, value=False, position=(10, 150), size=40, border_size=2,
                                 color_on='blue', color_off='lightblue')
    p.add_text("Tay Tu Dong (Auto)", position=(60, 160), font_size=12)

    p.add_checkbox_button_widget(btn_undo, value=False, position=(10, 100), size=40, border_size=2,
                                 color_on='orange', color_off='gold')
    p.add_text("Hoan Tac (Undo)", position=(60, 110), font_size=12)

    p.add_checkbox_button_widget(btn_save, value=False, position=(10, 50), size=40, border_size=2,
                                 color_on='green', color_off='lightgreen')
    p.add_text("Luu & Thoat (Save)", position=(60, 60), font_size=12)

    # Chức năng Tẩy Thủ Công (Manual Erase) bằng chuột
    # Dùng frustum picking để tránh lỗi với button widgets
    def on_rectangle_pick(selection):
        print(f"[DEBUG] Callback triggered! selection type: {type(selection)}")
        try:
            # Lấy toạ độ hình chữ nhật 2D đã vẽ (tính bằng PIXELS)
            x0, y0, x1, y1 = selection.viewport
            print(f"[DEBUG] viewport (pixels): x0={x0:.1f}, y0={y0:.1f}, x1={x1:.1f}, y1={y1:.1f}")
            xmin, xmax = min(x0, x1), max(x0, x1)
            ymin, ymax = min(y0, y1), max(y0, y1)

            # Chiếu tất cả đỉnh mesh ra toạ độ PIXELS trên màn hình
            renderer = p.renderer
            pts_world = state.current_mesh.points
            screen_pts = np.zeros((len(pts_world), 2))
            for i, pt in enumerate(pts_world):
                renderer.SetWorldPoint(pt[0], pt[1], pt[2], 1.0)
                renderer.WorldToDisplay()
                dp = renderer.GetDisplayPoint()
                screen_pts[i] = [dp[0], dp[1]]

            # Tìm các đỉnh nằm trong hình chữ nhật
            inside_mask = (
                (screen_pts[:, 0] >= xmin) & (screen_pts[:, 0] <= xmax) &
                (screen_pts[:, 1] >= ymin) & (screen_pts[:, 1] <= ymax)
            )

            if not inside_mask.any():
                print("Không có đỉnh nào trong vùng chọn!")
                return

            # Xóa các cell mà bất kỳ đỉnh nào nằm trong vùng chọn
            faces = state.current_mesh.faces.reshape(-1, 4)[:, 1:4]
            cell_inside = np.any(inside_mask[faces], axis=1)
            cells_to_remove = np.where(cell_inside)[0].tolist()

            if len(cells_to_remove) == 0:
                print("Không có cell nào bị khoanh!")
                return

            print(f"Đang xóa {len(cells_to_remove)} tam giác trong vùng chọn...")
            new_mesh = state.current_mesh.remove_cells(cells_to_remove)
            state.update_mesh(new_mesh)
            print("Xoá thủ công thành công!")
        except Exception as e:
            import traceback
            print(f"Lỗi khi xóa: {e}")
            traceback.print_exc()
            
    # Bật chế độ chọn hình chữ nhật. Ấn 'r' để kích hoạt, rồi kéo chuột.
    p.enable_rectangle_picking(
        callback=on_rectangle_pick,
        show_message="Nhan 'r' + keo chuot de khoanh vung rac can xoa",
        style='wireframe'
    )

    print("Mở cửa sổ Mesh Cleaner...")
    p.show()

if __name__ == "__main__":
    main()
