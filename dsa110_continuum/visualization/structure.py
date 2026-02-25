"""Generate a Mermaid flowchart diagram for any directory structure.

This module provides tools to analyze directory structures and generate
professional visual diagrams showing:
- Python package organization with color coding
- Module relationships with clear data flow indicators
- Config file usage (YAML, TOML, JSON) and which modules load them
- Directory structure with improved visual hierarchy and spatial organization
"""

import ast
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Optional

from .mermaid import MermaidRenderer

logger = logging.getLogger(__name__)


class ConfigUsageAnalyzer:
    """Analyze Python files to find config file usage patterns."""

    # Config file extensions to track
    CONFIG_EXTENSIONS = {".yaml", ".yml", ".toml", ".json", ".ini", ".cfg"}

    # Patterns to detect config loading
    CONFIG_LOADING_PATTERNS = [
        r"yaml\.safe_load\s*\(",
        r"yaml\.load\s*\(",
        r"\.from_yaml\s*\(",
        r"toml\.load\s*\(",
        r"toml\.loads\s*\(",
        r"json\.load\s*\(",
        r"json\.loads\s*\(",
        r"load_template\s*\(",
        r"load_config\s*\(",
    ]

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.config_files = {}  # config_path -> Set[module_path]
        self.config_usage = defaultdict(set)  # module_path -> Set[config_path]

    def find_config_files(self) -> Dict[Path, Set[Path]]:
        """Find all config files in the directory tree."""
        config_files = {}
        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not self._is_ignored_dir(d)]

            for file in files:
                file_path = root_path / file
                if any(file_path.suffix == ext for ext in self.CONFIG_EXTENSIONS):
                    rel_path = file_path.relative_to(self.root_path)
                    config_files[rel_path] = file_path
        return config_files

    def _is_ignored_dir(self, dirname: str) -> bool:
        """Check if directory should be ignored."""
        ignore_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            "dist",
            "build",
            ".venv",
            "venv",
            ".coverage",
            ".idea",
            ".vscode",
        }
        return dirname in ignore_dirs or dirname.startswith(".")

    def analyze_python_file(self, py_file: Path) -> Set[Path]:
        """Analyze a Python file for config file usage."""
        config_refs = set()

        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Find string literals that look like config file paths
            config_refs.update(self._find_config_strings(content, py_file))

            # Find config loading function calls
            config_refs.update(self._find_config_calls(content, py_file))

        except Exception:
            # Skip files that can't be read or parsed
            pass

        return config_refs

    def _find_config_strings(self, content: str, py_file: Path) -> Set[Path]:
        """Find string literals that reference config files."""
        config_refs = set()

        # Pattern to match string literals ending in config extensions
        pattern = r'["\']([^"\']*\.(yaml|yml|toml|json|ini|cfg))["\']'
        matches = re.finditer(pattern, content, re.IGNORECASE)

        for match in matches:
            config_path_str = match.group(1)
            # Try to resolve relative paths
            config_path = self._resolve_config_path(config_path_str, py_file)
            if config_path:
                config_refs.add(config_path)

        return config_refs

    def _find_config_calls(self, content: str, py_file: Path) -> Set[Path]:
        """Find config loading function calls using AST parsing."""
        config_refs = set()

        try:
            tree = ast.parse(content, filename=str(py_file))
            visitor = ConfigCallVisitor(self.root_path, py_file)
            visitor.visit(tree)
            config_refs.update(visitor.config_refs)
        except SyntaxError:
            # Skip files with syntax errors
            pass

        return config_refs

    def _resolve_config_path(self, config_str: str, py_file: Path) -> Optional[Path]:
        """Resolve a config file path string to an actual Path."""
        # Handle absolute paths
        if os.path.isabs(config_str):
            config_path = Path(config_str)
            if config_path.exists() and self.root_path in config_path.parents:
                return config_path.relative_to(self.root_path)
            return None

        # Handle relative paths
        # Try relative to the Python file
        config_path = (py_file.parent / config_str).resolve()
        if config_path.exists() and self.root_path in config_path.parents:
            return config_path.relative_to(self.root_path)

        # Try relative to root
        config_path = (self.root_path / config_str).resolve()
        if config_path.exists() and self.root_path in config_path.parents:
            return config_path.relative_to(self.root_path)

        # Try common config directories
        for config_dir in ["config", "backend/config", "ops"]:
            config_path = (
                self.root_path / config_dir / Path(config_str).name
            ).resolve()
            if config_path.exists():
                return config_path.relative_to(self.root_path)

        return None

    def analyze(self, python_files: List[Path]) -> Dict[Path, Set[Path]]:
        """Analyze all Python files and return config usage mapping."""
        # First, find all config files
        all_config_files = self.find_config_files()

        # Analyze each Python file
        for py_file in python_files:
            if not py_file.exists():
                continue

            config_refs = self.analyze_python_file(py_file)
            rel_py_file = py_file.relative_to(self.root_path)

            for config_ref in config_refs:
                # Only track config files that actually exist
                if config_ref in all_config_files:
                    self.config_usage[rel_py_file].add(config_ref)
                    if config_ref not in self.config_files:
                        self.config_files[config_ref] = set()
                    self.config_files[config_ref].add(rel_py_file)

        return self.config_files


class ConfigCallVisitor(ast.NodeVisitor):
    """AST visitor to find config loading function calls."""

    def __init__(self, root_path: Path, py_file: Path):
        self.root_path = root_path
        self.py_file = py_file
        self.config_refs = set()

    def visit_Call(self, node):
        """Visit function call nodes."""
        # Check for yaml.safe_load(), yaml.load()
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ("safe_load", "load", "loads"):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "yaml"
                ):
                    self._extract_config_from_args(node.args)

        # Check for .from_yaml() calls
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "from_yaml":
                self._extract_config_from_args(node.args)

        # Check for load_template(), load_config() calls
        if isinstance(node.func, ast.Name):
            if node.func.id in ("load_template", "load_config"):
                self._extract_config_from_args(node.args)

        self.generic_visit(node)

    def _extract_config_from_args(self, args):
        """Extract config file path from function arguments."""
        for arg in args:
            if isinstance(arg, ast.Str):  # Python < 3.8
                self._resolve_string(arg.s)
            elif isinstance(arg, ast.Constant) and isinstance(
                arg.value, str
            ):  # Python >= 3.8
                self._resolve_string(arg.value)

    def _resolve_string(self, s: str):
        """Resolve a string to a config file path."""
        if any(
            s.endswith(ext)
            for ext in [".yaml", ".yml", ".toml", ".json", ".ini", ".cfg"]
        ):
            # Try to resolve relative to Python file
            config_path = (self.py_file.parent / s).resolve()
            if config_path.exists() and self.root_path in config_path.parents:
                self.config_refs.add(config_path.relative_to(self.root_path))
            else:
                # Try relative to root
                config_path = (self.root_path / s).resolve()
                if config_path.exists() and self.root_path in config_path.parents:
                    self.config_refs.add(config_path.relative_to(self.root_path))


class DirectoryAnalyzer:
    """Analyze directory structure and generate Mermaid diagram."""

    # Common file patterns to highlight
    KEY_FILES = {
        "config": [
            "pyproject.toml",
            "setup.py",
            "package.json",
            "tsconfig.json",
            "Makefile",
            "requirements.txt",
            "poetry.lock",
        ],
        "docs": ["README.md", "CHANGELOG.md", "LICENSE"],
        "scripts": ["*.sh", "*.py"],
    }

    # Common directory patterns
    IGNORE_DIRS = {
        ".git",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".coverage",
        ".mypy_cache",
        "dist",
        "build",
        ".venv",
        "venv",
        "*.egg-info",
        ".idea",
        ".vscode",
        ".DS_Store",
    }

    # Module categorization for spatial organization
    CORE_PIPELINE = ["conversion", "calibration", "imaging", "photometry", "catalog"]
    INFRASTRUCTURE = ["api", "dagster", "graphql", "database", "pipeline"]
    SUPPORTING = [
        "utils",
        "monitoring",
        "validation",
        "qa",
        "visualization",
        "evaluation",
        "diagnostics",
        "mosaic",
        "pointing",
        "processing",
        "rfi",
        "services",
        "webhooks",
        "workflow",
        "execution",
        "cache",
        "cli",
        "config",
        "docsearch",
        "simulation",
        "legacy-system",
    ]

    def __init__(
        self,
        root_path: str,
        max_depth: int = 4,
        max_items_per_group: int = 15,
        analyze_config_usage: bool = True,
    ):
        self.root_path = Path(root_path).resolve()
        self.max_depth = max_depth
        self.max_items_per_group = max_items_per_group
        self.analyze_config_usage = analyze_config_usage
        self.structure = defaultdict(list)
        self.python_packages = set()
        self.key_files = []
        self.subdirectories = defaultdict(list)
        self.config_analyzer = None
        self.config_files = {}  # config_path -> Set[module_path]
        self.config_usage = defaultdict(set)  # module_path -> Set[config_path]

    def is_ignored(self, path: Path) -> bool:
        """Check if path should be ignored."""
        name = path.name
        if name.startswith("."):
            return name in self.IGNORE_DIRS or any(
                name.startswith(prefix) for prefix in [".coverage", ".pytest"]
            )
        return name in self.IGNORE_DIRS

    def analyze(self):
        """Analyze the directory structure."""
        if not self.root_path.exists():
            raise ValueError(f"Directory does not exist: {self.root_path}")

        # Walk the directory tree
        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            try:
                rel_path = root_path.relative_to(self.root_path)
            except ValueError:
                continue
                
            depth = len(rel_path.parts)

            if depth > self.max_depth:
                dirs[:] = []  # Don't recurse deeper
                continue

            # Filter ignored directories
            dirs[:] = [d for d in dirs if not self.is_ignored(root_path / d)]

            # Check for Python packages
            if "__init__.py" in files:
                self.python_packages.add(rel_path)

            # Collect key files
            for file in files:
                if self.is_ignored(root_path / file):
                    continue
                file_path = rel_path / file
                if any(
                    file.endswith(ext)
                    for ext in [".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".sh"]
                ):
                    self.key_files.append(file_path)

            # Group subdirectories
            if depth == 1 and dirs:
                for d in dirs:
                    if not self.is_ignored(root_path / d):
                        self.subdirectories[rel_path].append(d)

        # Organize structure
        self._organize_structure()

        # Analyze config file usage if enabled
        if self.analyze_config_usage:
            self._analyze_config_usage()

    def _organize_structure(self):
        """Organize the discovered structure into logical groups."""
        # Group Python packages
        package_groups = defaultdict(list)
        for pkg_path in sorted(self.python_packages):
            parts = pkg_path.parts
            if len(parts) >= 2:
                # Group by parent directory (e.g., src/dsa110_contimg/api -> api)
                parent = parts[-2] if parts[-2] != "src" else parts[-1]
                package_groups[parent].append(pkg_path)
            else:
                package_groups["root"].append(pkg_path)

        self.structure["python_packages"] = package_groups

        # Group key files by type
        file_groups = defaultdict(list)
        for file_path in self.key_files:
            ext = file_path.suffix
            if ext in [".py"]:
                file_groups["python"].append(file_path)
            elif ext in [".ts", ".tsx"]:
                file_groups["typescript"].append(file_path)
            elif ext in [".js", ".jsx"]:
                file_groups["javascript"].append(file_path)
            elif ext == ".md":
                file_groups["docs"].append(file_path)
            elif ext == ".sh":
                file_groups["scripts"].append(file_path)

        self.structure["files"] = file_groups

        # Group top-level directories
        top_level = []
        for item in sorted(self.root_path.iterdir()):
            if item.is_dir() and not self.is_ignored(item):
                top_level.append(item.name)

        self.structure["top_level"] = top_level

    def _analyze_config_usage(self):
        """Analyze Python files for config file usage."""
        logger.info("Analyzing config file usage...")
        self.config_analyzer = ConfigUsageAnalyzer(self.root_path)

        # Get all Python files
        python_files = []
        for file_path in self.key_files:
            if file_path.suffix == ".py":
                full_path = self.root_path / file_path
                if full_path.exists():
                    python_files.append(full_path)

        # Analyze config usage
        self.config_files = self.config_analyzer.analyze(python_files)
        self.config_usage = self.config_analyzer.config_usage

        if self.config_files:
            logger.info(
                f"  Found {len(self.config_files)} config files used by {len(set().union(*self.config_files.values()))} modules"
            )

    def _sanitize_id(self, name: str) -> str:
        """Sanitize name for Mermaid node ID."""
        # Replace special characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        # Remove leading numbers
        sanitized = re.sub(r"^\d+", "", sanitized)
        return sanitized or "node"

    def _categorize_module(self, module_name: str) -> str:
        """Categorize a module into functional group."""
        if module_name in self.CORE_PIPELINE:
            return "core_pipeline"
        elif module_name in self.INFRASTRUCTURE:
            return "infrastructure"
        elif module_name in self.SUPPORTING:
            return "supporting"
        else:
            return "other"

    def generate_mermaid(self) -> str:
        """Generate Mermaid flowchart code with improved spatial organization."""
        root_name = self.root_path.name or "root"
        root_id = self._sanitize_id(root_name)

        # Use LR (left-right) layout for better horizontal flow and grouping
        lines = ["flowchart LR"]
        lines.append("    %% Optimized layout with compact, well-connected subgraphs")
        lines.append(
            "    %% Left-right flow with strategic grouping to reduce crossing lines"
        )

        # Add compact, well-connected subgraphs
        self._add_compact_subgraphs(lines)

        # Add relationships - CRITICAL for layout and connectivity
        self._add_relationships(lines)

        # Add config usage relationships with distinct styling
        if self.config_files:
            self._add_config_relationships(lines)

        # Connect all groups to prevent isolation
        self._connect_all_groups(lines)

        # Add enhanced styling
        self._add_styling(lines, root_id)

        # Add legend/notes
        self._add_legend(lines)

        return "\n".join(lines)

    def _add_functional_groups(self, lines: List[str]):
        """Add modules organized into functional subgraphs for better spatial distribution."""
        package_groups = self.structure.get("python_packages", {})

        if not package_groups:
            return

        # Find the main package directory (usually 'src' or root)
        main_pkg_base = None
        for pkg_path in sorted(self.python_packages):
            if "src" in pkg_path.parts:
                src_idx = pkg_path.parts.index("src")
                if len(pkg_path.parts) > src_idx + 1:
                    main_pkg_base = pkg_path.parts[src_idx + 1]
                    break

        # Organize modules by category
        categorized_modules = {
            "core_pipeline": {},
            "infrastructure": {},
            "supporting": {},
            "other": {},
        }

        for pkg_path in sorted(self.python_packages):
            if main_pkg_base and main_pkg_base in pkg_path.parts:
                parts = pkg_path.parts
                try:
                    base_idx = parts.index(main_pkg_base)
                    if len(parts) > base_idx + 1:
                        module_name = parts[base_idx + 1]
                        # Only include if it's a direct child (depth check)
                        if len(parts) == base_idx + 2:  # src/main_pkg/module
                            category = self._categorize_module(module_name)
                            if module_name not in categorized_modules[category]:
                                categorized_modules[category][module_name] = []
                            categorized_modules[category][module_name].append(pkg_path)
                except (ValueError, IndexError):
                    pass
            elif not main_pkg_base:
                # No main package base, use last part
                if pkg_path.parts:  # Fix: Check if parts exists
                    module_name = pkg_path.parts[-1]
                    category = self._categorize_module(module_name)
                    if module_name not in categorized_modules[category]:
                        categorized_modules[category][module_name] = []
                    categorized_modules[category][module_name].append(pkg_path)

        return categorized_modules

    def _build_categorized_modules(self):
        """Build categorized modules dictionary."""
        package_groups = self.structure.get("python_packages", {})

        if not package_groups:
            return {
                "core_pipeline": {},
                "infrastructure": {},
                "supporting": {},
                "other": {},
            }

        # Find the main package directory (usually 'src' or root)
        main_pkg_base = None
        for pkg_path in sorted(self.python_packages):
            if "src" in pkg_path.parts:
                src_idx = pkg_path.parts.index("src")
                if len(pkg_path.parts) > src_idx + 1:
                    main_pkg_base = pkg_path.parts[src_idx + 1]
                    break

        # Organize modules by category
        categorized_modules = {
            "core_pipeline": {},
            "infrastructure": {},
            "supporting": {},
            "other": {},
        }

        for pkg_path in sorted(self.python_packages):
            if main_pkg_base and main_pkg_base in pkg_path.parts:
                parts = pkg_path.parts
                try:
                    base_idx = parts.index(main_pkg_base)
                    if len(parts) > base_idx + 1:
                        module_name = parts[base_idx + 1]
                        # Only include if it's a direct child (depth check)
                        if len(parts) == base_idx + 2:  # src/main_pkg/module
                            category = self._categorize_module(module_name)
                            if module_name not in categorized_modules[category]:
                                categorized_modules[category][module_name] = []
                            categorized_modules[category][module_name].append(pkg_path)
                except (ValueError, IndexError):
                    pass
            elif not main_pkg_base:
                # No main package base, use last part
                if pkg_path.parts:  # Fix: Check if parts exists
                    module_name = pkg_path.parts[-1]
                    category = self._categorize_module(module_name)
                    if module_name not in categorized_modules[category]:
                        categorized_modules[category][module_name] = []
                    categorized_modules[category][module_name].append(pkg_path)

        return categorized_modules

    def _add_compact_subgraphs(self, lines: List[str]):
        """Add compact, well-connected subgraphs with strategic grouping."""
        categorized_modules = self._build_categorized_modules()

        # Core Pipeline subgraph - compact horizontal flow (left side)
        if categorized_modules["core_pipeline"]:
            lines.append('    subgraph CorePipeline["Core Pipeline"]')
            lines.append("        direction LR")
            pipeline_order = [
                "conversion",
                "calibration",
                "imaging",
                "photometry",
                "catalog",
            ]
            for module_name in pipeline_order:
                if module_name in categorized_modules["core_pipeline"]:
                    self._add_module_node(
                        lines,
                        module_name,
                        categorized_modules["core_pipeline"][module_name],
                    )
            lines.append("    end")

        # Infrastructure subgraph - compact grouping (middle)
        if categorized_modules["infrastructure"]:
            lines.append('    subgraph Infrastructure["Infrastructure"]')
            lines.append("        direction LR")
            infra_order = ["api", "dagster", "graphql", "database", "pipeline"]
            for module_name in infra_order:
                if module_name in categorized_modules["infrastructure"]:
                    self._add_module_node(
                        lines,
                        module_name,
                        categorized_modules["infrastructure"][module_name],
                    )
            lines.append("    end")

        # Supporting modules subgraph - compact grouping (right side, key modules only)
        if categorized_modules["supporting"]:
            lines.append('    subgraph Supporting["Supporting"]')
            lines.append("        direction LR")
            # Show only most important supporting modules to keep compact
            key_supporting = ["utils", "monitoring", "validation", "qa"]
            for module_name in key_supporting:
                if module_name in categorized_modules["supporting"]:
                    self._add_module_node(
                        lines,
                        module_name,
                        categorized_modules["supporting"][module_name],
                    )
            lines.append("    end")

        # Other modules - only if few and meaningful (compact)
        if categorized_modules["other"] and len(categorized_modules["other"]) <= 4:
            lines.append('    subgraph Other["Other"]')
            lines.append("        direction LR")
            for module_name in sorted(categorized_modules["other"].keys())[:4]:
                self._add_module_node(
                    lines, module_name, categorized_modules["other"][module_name]
                )
            lines.append("    end")

    def _add_module_node(
        self, lines: List[str], module_name: str, packages: List[Path]
    ):
        """Add a single module node with clean, simple label and rounded corners."""
        # Clean label: just module name (no file clutter)
        # Use rounded rectangle shape for softer appearance
        desc = module_name

        node_id = self._sanitize_id(module_name)
        # Store for relationship mapping
        if not hasattr(self, "_node_ids"):
            self._node_ids = {}
        self._node_ids[module_name] = node_id
        # Use rounded rectangle syntax for better visual appearance
        lines.append(f'        {node_id}("{desc}")')

    def _add_config_files(self, lines: List[str]):
        """Add config file nodes to the diagram."""
        if not self.config_files:
            return

        lines.append('    subgraph ConfigFiles["Config Files"]')
        lines.append("        direction TB")

        # Group config files by directory
        config_by_dir = defaultdict(list)
        for config_path in sorted(self.config_files.keys()):
            if len(config_path.parts) > 1:
                dir_name = config_path.parts[0]
                config_by_dir[dir_name].append(config_path)
            else:
                config_by_dir["root"].append(config_path)

        # Add config files (simplified labels with rounded corners)
        for dir_name, configs in sorted(config_by_dir.items()):
            if len(configs) <= 4:
                for config_path in configs:
                    display_name = config_path.name
                    node_id = self._sanitize_id(
                        f"config_{config_path.stem.replace('.', '_')}"
                    )
                    lines.append(f'        {node_id}("{display_name}")')
            else:
                node_id = self._sanitize_id(f"config_{dir_name}")
                file_list = ", ".join([c.name for c in configs[:3]])
                if len(configs) > 3:
                    file_list += f"<br/>(+{len(configs) - 3} more)"
                lines.append(f'        {node_id}("{dir_name}/<br/>{file_list}")')

        lines.append("    end")

    def _add_other_directories(self, lines: List[str]):
        """Add other important directories with rounded corners."""
        # Add tests directory
        test_dirs = [
            d for d in self.structure.get("top_level", []) if "test" in d.lower()
        ]
        if test_dirs:
            for test_dir in test_dirs[:3]:
                node_id = self._sanitize_id(test_dir)
                lines.append(f'    {node_id}("{test_dir}/")')

        # Add docs directory
        if "docs" in self.structure.get("top_level", []):
            lines.append('    Docs("docs/")')

        # Add scripts directory
        if "scripts" in self.structure.get("top_level", []):
            lines.append('    Scripts("scripts/")')

    def _add_config_relationships(self, lines: List[str]):
        """Add relationships between config files and modules that use them."""
        if not self.config_files:
            return

        # Build module name to node ID mapping
        module_to_id = {}
        package_groups = self.structure.get("python_packages", {})
        for group_name, packages in package_groups.items():
            for pkg in packages:
                if pkg.parts:  # Fix: Check if parts exists
                    module_name = pkg.parts[-1]
                    module_to_id[module_name] = self._sanitize_id(module_name)

        # Also check for module paths
        for module_path in self.config_usage.keys():
            parts = module_path.parts
            if len(parts) >= 2:
                if "src" in parts:
                    src_idx = parts.index("src")
                    if len(parts) > src_idx + 2:
                        module_name = parts[src_idx + 2]
                        if module_name not in module_to_id:
                            module_to_id[str(module_path)] = self._sanitize_id(
                                module_name
                            )
                else:
                    module_name = parts[-2] if len(parts) >= 2 else parts[-1]
                    if module_name not in module_to_id:
                        module_to_id[str(module_path)] = self._sanitize_id(module_name)

        # Add relationships: config files -> modules
        for config_path, using_modules in self.config_files.items():
            config_id = self._sanitize_id(
                f"config_{config_path.stem.replace('.', '_')}"
            )

            for module_path in using_modules:
                module_id = None
                parts = module_path.parts

                if len(parts) >= 2:
                    if "src" in parts:
                        src_idx = parts.index("src")
                        if len(parts) > src_idx + 2:
                            module_name = parts[src_idx + 2]
                            module_id = module_to_id.get(module_name)
                    else:
                        module_name = parts[-2] if len(parts) >= 2 else parts[-1]
                        module_id = module_to_id.get(module_name)

                if not module_id:
                    module_id = module_to_id.get(str(module_path))

                if not module_id and parts:
                    module_name = parts[-1].replace(".py", "")
                    module_id = module_to_id.get(module_name)

                if module_id:
                    lines.append(f"    {config_id} -.-> {module_id}")

    def _add_relationships(self, lines: List[str]):
        """Add relationships between modules with improved styling."""
        lines.append(
            "    %% Module relationships - organize layout through connections"
        )

        # Use stored node IDs from _add_compact_subgraphs
        organized = getattr(self, "_node_ids", {})
        if not organized:
            # Fallback: build mapping
            package_groups = self.structure.get("python_packages", {})
            for group_name, packages in package_groups.items():
                for pkg in packages:
                    if pkg.parts:
                        module_name = pkg.parts[-1]
                        organized[module_name] = self._sanitize_id(module_name)

        # Connect pipeline stages with thicker lines for main flow (horizontal in LR layout)
        # Use clean arrow labels for better readability
        pipeline_order = [
            "conversion",
            "calibration",
            "imaging",
            "photometry",
            "catalog",
        ]
        prev = None
        for stage in pipeline_order:
            if stage in organized:
                node_id = organized[stage]
                if prev:
                    lines.append(f"    {prev} ==>|→| {node_id}")
                prev = node_id

        # Connect Infrastructure to Core Pipeline (short, direct connection)
        if "pipeline" in organized and "conversion" in organized:
            lines.append(f"    {organized['pipeline']} --> {organized['conversion']}")

        # Connect API to database (within Infrastructure group)
        if "api" in organized and "database" in organized:
            lines.append(f"    {organized['api']} --> {organized['database']}")

        # Connect utils to core modules (dashed for utility relationships)
        if "utils" in organized:
            utils_id = organized["utils"]
            for module in ["conversion", "calibration", "imaging"]:
                if module in organized:
                    lines.append(f"    {utils_id} -.-> {organized[module]}")

        # Connect monitoring/validation to pipeline (short connections)
        for module in ["monitoring", "validation", "qa"]:
            if module in organized and "pipeline" in organized:
                lines.append(f"    {organized[module]} -.-> {organized['pipeline']}")

    def _add_styling(self, lines: List[str], root_id: str):
        """Add cohesive, professional color styling with improved contrast using classDef."""
        lines.append("    %% Cohesive color palette - professional blue/gray scheme")
        lines.append("    %% Using classDef for cleaner, more maintainable styling")

        # Define color classes with rounded corners for softer appearance
        lines.append(
            "    classDef corePipeline fill:#3B82F6,stroke:#1E40AF,stroke-width:2.5px,color:#FFFFFF,stroke-dasharray:0"
        )
        lines.append(
            "    classDef infrastructure fill:#0284C7,stroke:#0C4A6E,stroke-width:2.5px,color:#FFFFFF,stroke-dasharray:0"
        )
        lines.append(
            "    classDef supporting fill:#6B7280,stroke:#374151,stroke-width:2px,color:#FFFFFF,stroke-dasharray:0"
        )
        lines.append(
            "    classDef other fill:#9CA3AF,stroke:#4B5563,stroke-width:1.5px,color:#1F2937,stroke-dasharray:0"
        )
        lines.append(
            "    classDef config fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#78350F,stroke-dasharray:0"
        )
        lines.append(
            "    classDef directory fill:#F3F4F6,stroke:#9CA3AF,stroke-width:1.5px,color:#374151,stroke-dasharray:0"
        )

        # Subgraph styling - compact, professional backgrounds with subtle shadows
        lines.append(
            "    style CorePipeline fill:#E8F4FD,stroke:#3B82F6,stroke-width:2.5px,stroke-dasharray:0"
        )
        lines.append(
            "    style Infrastructure fill:#F0F9FF,stroke:#0284C7,stroke-width:2.5px,stroke-dasharray:0"
        )
        lines.append(
            "    style Supporting fill:#F9FAFB,stroke:#6B7280,stroke-width:2px,stroke-dasharray:0"
        )
        lines.append(
            "    style Other fill:#F9FAFB,stroke:#9CA3AF,stroke-width:1.5px,stroke-dasharray:0"
        )

    def _connect_all_groups(self, lines: List[str]):
        """Connect all subgraphs to prevent isolation and improve spatial distribution."""
        lines.append(
            "    %% Connect groups to ensure cohesive layout and reduce fragmentation"
        )

        organized = getattr(self, "_node_ids", {})
        if not organized:
            return

        # Connect Core Pipeline to Infrastructure (main connection)
        if "conversion" in organized and "pipeline" in organized:
            lines.append(f"    {organized['conversion']} -.-> {organized['pipeline']}")

        # Connect Infrastructure to Supporting (bridge connection)
        if "pipeline" in organized and "utils" in organized:
            lines.append(f"    {organized['pipeline']} -.-> {organized['utils']}")

        # Connect Supporting to Other (if Other exists)
        other_modules = [
            m
            for m in organized.keys()
            if m in ["batch", "commands", "routes", "middleware"]
        ]
        if "utils" in organized and other_modules:
            first_other = other_modules[0]
            lines.append(f"    {organized['utils']} -.-> {organized[first_other]}")

        # Connect API to supporting modules (cross-group connections)
        if "api" in organized:
            api_id = organized["api"]
            for module in ["monitoring", "validation"]:
                if module in organized:
                    lines.append(f"    {api_id} -.-> {organized[module]}")

        # Apply classes ONLY to modules actually shown in the diagram
        # Use stored node IDs from _add_compact_subgraphs to avoid assigning classes to hidden modules
        organized = getattr(self, "_node_ids", {})
        if not organized:
            # Fallback: build mapping
            package_groups = self.structure.get("python_packages", {})
            for group_name, packages in package_groups.items():
                for pkg in packages:
                    if pkg.parts:
                        module_name = pkg.parts[-1]
                        organized[module_name] = self._sanitize_id(module_name)

        # Core Pipeline modules (only those shown)
        for module in self.CORE_PIPELINE:
            if module in organized:
                lines.append(f"    class {organized[module]} corePipeline")

        # Infrastructure modules (only those shown)
        for module in self.INFRASTRUCTURE:
            if module in organized:
                lines.append(f"    class {organized[module]} infrastructure")

        # Supporting modules (only those shown)
        key_supporting = ["utils", "monitoring", "validation", "qa"]
        for module in key_supporting:
            if module in organized:
                lines.append(f"    class {organized[module]} supporting")

        # Other modules (only those shown)
        categorized_modules = self._build_categorized_modules()
        if categorized_modules["other"] and len(categorized_modules["other"]) <= 4:
            for module_name in sorted(categorized_modules["other"].keys())[:4]:
                if module_name in organized:
                    lines.append(f"    class {organized[module_name]} other")

        # Config files
        if self.config_files:
            lines.append(
                "    style ConfigFiles fill:#FEF3C7,stroke:#D97706,stroke-width:2px"
            )
            for config_path in self.config_files.keys():
                config_id = self._sanitize_id(
                    f"config_{config_path.stem.replace('.', '_')}"
                )
                lines.append(f"    class {config_id} config")

        # Other directories
        for dir_name in ["Docs", "Scripts", "tests"]:
            node_id = self._sanitize_id(dir_name)
            if node_id in [
                self._sanitize_id(d) for d in self.structure.get("top_level", [])
            ]:
                lines.append(f"    class {node_id} directory")

    def _is_light_color(self, hex_color: str) -> bool:
        """Check if a hex color is light (for text color selection)."""
        # Remove # if present
        hex_color = hex_color.lstrip("#")
        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        # Calculate luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return luminance > 0.6

    def _add_legend(self, lines: List[str]):
        """Add visual notes explaining the diagram."""
        lines.append("")
        lines.append("    %% Visual Guide & Legend")
        lines.append("    %% ====================")
        lines.append(
            "    %% Main Data Flow: Thick arrows (==>) show core pipeline progression"
        )
        lines.append(
            "    %% Dependencies: Solid arrows (-->) show direct module dependencies"
        )
        lines.append(
            "    %% Utilities: Dashed arrows (-.->) show supporting/utility relationships"
        )
        lines.append(
            "    %% Layout: Left-to-right (LR) flow matches pipeline execution order"
        )
        lines.append(
            "    %% Grouping: Subgraphs organize modules by functional purpose"
        )
        lines.append(
            "    %% Colors: Blue (core), Cyan (infrastructure), Gray (supporting)"
        )


def generate_structure_diagram(directory_path: str, output_file: str) -> bool:
    """Generate structure diagram for a directory.

    Parameters
    ----------
    directory_path : str
        Path to the directory to analyze
    output_file : str
        Path to save the generated SVG

    Returns
    -------
    bool
        True if successful, False otherwise

    Example
    -------
    >>> from dsa110_contimg.core.visualization import generate_structure_diagram
    >>> generate_structure_diagram("/path/to/source", "output_diagram.svg")
    """
    try:
        # Analyze directory
        logger.info(f"Analyzing directory: {directory_path}")
        analyzer = DirectoryAnalyzer(directory_path)
        analyzer.analyze()

        # Generate Mermaid code
        logger.info("Generating Mermaid diagram...")
        mermaid_code = analyzer.generate_mermaid()

        # Optionally save Mermaid source
        mermaid_file = str(Path(output_file).with_suffix(".mmd"))
        with open(mermaid_file, "w", encoding="utf-8") as f:
            f.write(mermaid_code)
        logger.info(f"Mermaid source saved to: {mermaid_file}")

        # Render to SVG
        renderer = MermaidRenderer()
        return renderer.render(mermaid_code, output_file)

    except Exception as e:
        logger.error(f"Error generating diagram: {e}")
        return False
