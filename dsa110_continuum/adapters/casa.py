"""
CASA 6 Adapter Layer.

This module encapsulates all interactions with the `casatools` and `casaconfig` libraries.
It serves as an Anti-Corruption Layer (ACL) to protect the core application domain
from direct dependencies on the CASA 6 modular structure.

Usage:
    from dsa110_contimg.core.adapters.casa import casa_adapter
    
    # Get a tool instance
    qa = casa_adapter.quanta()
    
    # Check configuration
    if casa_adapter.is_available:
        path = casa_adapter.datapath
"""

import os
import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

class CasaAdapter:
    """Singleton adapter for CASA 6 tools and configuration."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CasaAdapter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._available = False
        self._casatools = None
        self._casaconfig = None
        self._casa_config_module = None
        
        try:
            import casatools
            self._casatools = casatools
            self._available = True
        except ImportError:
            logger.warning("casatools not found. CASA integration disabled.")
            
        # Try to load configuration
        try:
            import casaconfig
            self._casaconfig = casaconfig
            
            # Robustly access the config submodule
            try:
                from casaconfig import config
                self._casa_config_module = config
            except ImportError:
                self._casa_config_module = getattr(casaconfig, "config", None)
                
        except ImportError:
            logger.debug("casaconfig not found.")
            
        self._initialized = True

    @property
    def is_available(self) -> bool:
        """Return True if casatools is installed."""
        return self._available

    @property
    def version(self) -> str:
        """Return CASA version string or 'unknown'."""
        if not self._available:
            return "unknown"
        
        try:
            v = self._casatools.version()
            if isinstance(v, (list, tuple)):
                return ".".join(str(x) for x in v)
            return str(v)
        except Exception:
            return "unknown"

    @property
    def datapath(self) -> Optional[List[str]]:
        """Return list of data paths from casaconfig or None."""
        if self._casa_config_module:
            return getattr(self._casa_config_module, "datapath", None)
        return None

    def configure_runtime(self, disable_auto_updates: bool = True) -> None:
        """Apply runtime configuration settings."""
        if self._casa_config_module and disable_auto_updates:
            try:
                self._casa_config_module.auto_update_rules = False
                self._casa_config_module.measures_auto_update = False
                self._casa_config_module.data_auto_update = False
            except AttributeError:
                pass

    # --- Tool Factories ---

    def quanta(self) -> Any:
        """Return a new casatools.quanta instance."""
        if not self._available:
            raise RuntimeError("CASA tools not available")
        return self._casatools.quanta()

    def table(self) -> Any:
        """Return a new casatools.table instance."""
        if not self._available:
            raise RuntimeError("CASA tools not available")
        return self._casatools.table()
        
    def measures(self) -> Any:
        """Return a new casatools.measures instance."""
        if not self._available:
            raise RuntimeError("CASA tools not available")
        return self._casatools.measures()

    def casalog(self) -> Any:
        """Return the global casatools.casalog instance."""
        if not self._available:
            raise RuntimeError("CASA tools not available")
        # In CASA 6, casalog is often a top-level import from casatools
        # It is NOT always available as casatools.casalog
        try:
            from casatools import casalog
            return casalog
        except ImportError:
            # Fallback if it is attached to the module
            if hasattr(self._casatools, "casalog"):
                return self._casatools.casalog
            raise RuntimeError("Could not find casalog in casatools")

# Global singleton instance
casa_adapter = CasaAdapter()
