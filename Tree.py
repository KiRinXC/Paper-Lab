import os
from pathlib import Path


def print_directory_tree(root_dir, indent='', max_depth=None, current_depth=0):
    """
    递归打印目录树结构

    参数:
        root_dir: 根目录路径
        indent: 缩进字符串
        max_depth: 最大遍历深度(None表示无限制)
        current_depth: 当前深度
    """
    if max_depth is not None and current_depth > max_depth:
        return

    # 获取目录下的所有条目并排序
    try:
        entries = sorted(os.listdir(root_dir))
    except PermissionError:
        print(indent + "⛔ [权限被拒绝] " + os.path.basename(root_dir))
        return
    except FileNotFoundError:
        print(indent + "❌ [目录不存在] " + os.path.basename(root_dir))
        return

    for i, entry in enumerate(entries):
        path = os.path.join(root_dir, entry)
        is_last = i == len(entries) - 1

        if os.path.isdir(path):
            # 打印目录
            print(indent + ('└── ' if is_last else '├── ') + '📁 ' + entry)
            # 递归打印子目录
            new_indent = indent + ('    ' if is_last else '│   ')
            print_directory_tree(
                path,
                new_indent,
                max_depth,
                current_depth + 1
            )
        else:
            # 打印文件
            print(indent + ('└── ' if is_last else '├── ') + '📄 ' + entry)


def get_directory_structure(root_dir, max_depth=None):
    """
    获取目录结构的字符串表示

    参数:
        root_dir: 根目录路径
        max_depth: 最大遍历深度
    返回:
        目录结构的字符串表示
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        return f"目录不存在: {root_dir}"

    lines = []
    prefix = []

    def add_directory(directory, prefix):
        """递归添加目录内容"""
        nonlocal lines
        entries = sorted(directory.iterdir())
        entries_count = len(entries)

        for i, entry in enumerate(entries):
            is_last = i == entries_count - 1
            if entry.is_dir():
                lines.append(''.join(prefix) + ('└── ' if is_last else '├── ') + '📁 ' + entry.name)
                if max_depth is None or len(prefix) // 4 < max_depth:
                    prefix.append('    ' if is_last else '│   ')
                    add_directory(entry, prefix)
                    prefix.pop()
            else:
                lines.append(''.join(prefix) + ('└── ' if is_last else '├── ') + '📄 ' + entry.name)

    lines.append('📂 ' + str(root_path))
    add_directory(root_path, prefix)
    return '\n'.join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='输出目录结构')
    parser.add_argument('directory', nargs='?', default='.', help='要显示的目录路径(默认为当前目录)')
    parser.add_argument('-d', '--depth', type=int, help='最大显示深度')
    parser.add_argument('-o', '--output', help='输出到文件')

    args = parser.parse_args()

    # 获取目录结构
    dir_structure = get_directory_structure(args.directory, args.depth)

    # 输出到文件或控制台
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(dir_structure)
        print(f"目录结构已保存到: {args.output}")
    else:
        print(dir_structure)

'''
 python Tree.py D:\Code\Demo\Paper-Lab\Dataset\FEMTOBearingDataSet -d 1 -o dir.txt
'''