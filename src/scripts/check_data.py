import argparse
from pathlib import Path
from data import ICD10HierarchyLoader, ICD10Node

def main(): 
    
    parser = argparse.ArgumentParser(description="Explore ICD10 Hierarchy Loader")
    parser.add_argument(
        "--xml-path",
        type=str,
        default="../data/FY24-CMS-1785-F-ICD-10-Table-Index/icd10cm_tabular_2024.xml",
        help="Path to ICD10 XML file"
    )
    parser.add_argument(
        "--code",
        type=str,
        default=None,
        help="ICD10 code to explore (e.g., I10, E11.29)"
    )
    parser.add_argument(
        "--show-children",
        type=str,
        default=None,
        help="Show children of a specific code"
    )
    parser.add_argument(
        "--show-path",
        type=str,
        default=None,
        help="Show path from root to a specific code"
    )
    parser.add_argument(
        "--sample-codes",
        type=int,
        default=10,
        help="Number of sample codes to display"
    )
    parser.add_argument(
        "--leaf-only",
        action="store_true",
        help="Show only leaf codes in samples"
    )
    
    args = parser.parse_args()
    
    # Load hierarchy
    print("=" * 70)
    print("ICD10 Hierarchy Loader - Explorer")
    print("=" * 70)
    print(f"\nLoading hierarchy from: {args.xml_path}")
    
    xml_path = Path(args.xml_path)
    if not xml_path.exists():
        print(f"Error: XML file not found at {xml_path}")
        print("Please provide a valid path to the ICD10 XML file.")
        exit(1)
    
    loader = ICD10HierarchyLoader(str(xml_path))
    root_node = loader.load()
    
    print("✓ Hierarchy loaded successfully!")
    print()
    
    # Display basic statistics
    all_codes = loader.get_all_codes()
    leaf_codes = loader.get_leaf_codes()
    
    print("=" * 70)
    print("Hierarchy Statistics")
    print("=" * 70)
    print(f"Total codes: {len(all_codes)}")
    print(f"Leaf codes: {len(leaf_codes)}")
    print(f"Non-leaf codes: {len(all_codes) - len(leaf_codes)}")
    print()
    
    # Show sample codes
    print("=" * 70)
    print(f"Sample Codes ({'Leaf Only' if args.leaf_only else 'All'})")
    print("=" * 70)
    sample_codes = leaf_codes[:args.sample_codes] if args.leaf_only else all_codes[1:args.sample_codes+1]  # Skip ROOT
    for code in sample_codes:
        node = loader.get_node(code)
        if node:
            print(f"\nCode: {node.code}")
            print(f"  Description: {node.description}")
            if node.laymen_definition:
                print(f"  Laymen Definition: {node.laymen_definition[:100]}...")
            print(f"  Is Leaf: {node.is_leaf}")
            print(f"  Number of Children: {len(node.children)}")
            if node.parent and node.parent.code != "ROOT":
                print(f"  Parent: {node.parent.code} - {node.parent.description}")
    print()
    
    # Explore specific code
    if args.code:
        print("=" * 70)
        print(f"Exploring Code: {args.code}")
        print("=" * 70)
        node = loader.get_node(args.code)
        if node:
            print(f"\nCode: {node.code}")
            print(f"Description: {node.description}")
            if node.laymen_definition:
                print(f"Laymen Definition: {node.laymen_definition}")
            if node.inclusion_terms:
                print(f"Inclusion Terms: {', '.join(node.inclusion_terms)}")
            print(f"Is Leaf: {node.is_leaf}")
            print(f"Number of Children: {len(node.children)}")
            if node.parent:
                print(f"Parent: {node.parent.code} - {node.parent.description}")
            if node.children:
                print(f"\nChildren ({len(node.children)}):")
                for child in node.children[:10]:  # Show first 10 children
                    print(f"  - {child.code}: {child.description}")
                if len(node.children) > 10:
                    print(f"  ... and {len(node.children) - 10} more")
        else:
            print(f"Code '{args.code}' not found in hierarchy")
        print()
    
    # Show children of a code
    if args.show_children:
        print("=" * 70)
        print(f"Children of Code: {args.show_children}")
        print("=" * 70)
        children = loader.get_children(args.show_children)
        if children:
            print(f"\nFound {len(children)} children:\n")
            for child in children:
                print(f"  {child.code}: {child.description}")
                if child.is_leaf:
                    print(f"    (Leaf node)")
                else:
                    print(f"    (Has {len(child.children)} children)")
        else:
            node = loader.get_node(args.show_children)
            if node:
                print(f"Code '{args.show_children}' has no children (leaf node)")
            else:
                print(f"Code '{args.show_children}' not found")
        print()
    
    # Show path to root
    if args.show_path:
        print("=" * 70)
        print(f"Path from Root to Code: {args.show_path}")
        print("=" * 70)
        path = loader.get_path_to_root(args.show_path)
        if path:
            print(f"\nPath (from root to {args.show_path}):\n")
            for i, node in enumerate(path):
                indent = "  " * i
                print(f"{indent}{node.code}: {node.description}")
        else:
            print(f"Code '{args.show_path}' not found in hierarchy")
        print()
    
    # Show hierarchy structure example
    print("=" * 70)
    print("Hierarchy Structure Example")
    print("=" * 70)
    print("\nShowing first few levels of hierarchy:\n")
    
    def print_tree(node: ICD10Node, level: int = 0, max_level: int = 3, max_children: int = 3):
        """Print tree structure."""
        if level > max_level:
            return
        indent = "  " * level
        print(f"{indent}{node.code}: {node.description[:60]}")
        if node.children and level < max_level:
            for child in node.children[:max_children]:
                print_tree(child, level + 1, max_level, max_children)
            if len(node.children) > max_children:
                print(f"{indent}  ... and {len(node.children) - max_children} more children")
    
    # Start from root and show first few chapters
    if root_node.children:
        for chapter in root_node.children[:3]:  # Show first 3 chapters
            print_tree(chapter, level=0, max_level=2, max_children=2)
            print()
    
    print("=" * 70)
    print("Exploration Complete!")
    print("=" * 70)
    print("\nUsage examples:")
    print("  uv run scripts/check_data.py --code I10")
    print("  uv run scripts/check_data.py --show-children I10")
    print("  uv run scripts/check_data.py --show-path E11.29")
    print("  uv run scripts/check_data.py --leaf-only --sample-codes 20")
    print()


if __name__ == "__main__":
    main()
