#include <iostream>
#include <fstream>

#include "PluginManager.h"

//#ifndef SRC_DIR
//#error "SRC_DIR undefined"
//#endif
//#ifndef LIB_DIR
//#error "LIB_DIR undefined"
//#endif

#define STR_EXPAND(x) #x
#define STR(x) STR_EXPAND(x)

namespace edmplugin {
  PluginManager::PluginManager() {
    std::ifstream pluginMap("/models/identity_fp32/1/plugins.txt");
    std::string plugin, library;
    while (pluginMap >> plugin >> library) {
      std::cout << "plugin " << plugin << " in " << library << std::endl;
      pluginToLibrary_[plugin] = library;
    }
  }

  SharedLibrary const& PluginManager::load(std::string const& pluginName) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    std::cout << "--> here 1 " << pluginName << std::endl;
    auto libName = pluginToLibrary_.at(pluginName);
    std::cout << "--> here 2 " << std::endl;
    auto found = loadedPlugins_.find(libName);
    std::cout << "--> here 3 " << std::endl;
    if (found == loadedPlugins_.end()) {
      auto ptr = std::make_shared<SharedLibrary>("/models/identity_fp32/1/" + libName);
      loadedPlugins_[libName] = ptr;
      return *ptr;
    }
    std::cout << "--> here 4 " << std::endl;
    return *(found->second);
  }
}  // namespace edmplugin
