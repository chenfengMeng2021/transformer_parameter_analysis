# AI Block Research

<div align="center">
  <h2>Transformer Parameter Analysis</h2>
  <p>A research toolkit for downloading, reading, and analyzing large language model parameters</p>
  
  <div style="margin: 20px 0;">
    <button onclick="showLanguage('en')" id="btn-en" style="padding: 10px 20px; margin: 0 10px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">🇺🇸 English</button>
    <button onclick="showLanguage('zh')" id="btn-zh" style="padding: 10px 20px; margin: 0 10px; background-color: #6c757d; color: white; border: none; border-radius: 5px; cursor: pointer;">🇨🇳 中文</button>
  </div>
</div>

---

<div id="content-en" style="display: block;">
  <!-- English content will be loaded here -->
</div>

<div id="content-zh" style="display: none;">
  <!-- Chinese content will be loaded here -->
</div>

<script>
// Simple markdown parser for basic formatting
function simpleMarkdownParser(text) {
  return text
    // Headers
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    // Bold
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    // Code blocks
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Lists
    .replace(/^\* (.*$)/gim, '<li>$1</li>')
    .replace(/^- (.*$)/gim, '<li>$1</li>')
    // Line breaks
    .replace(/\n/g, '<br>');
}

// Function to show selected language content
function showLanguage(lang) {
  // Hide all content divs
  document.getElementById('content-en').style.display = 'none';
  document.getElementById('content-zh').style.display = 'none';
  
  // Show selected language content
  document.getElementById('content-' + lang).style.display = 'block';
  
  // Update button styles
  document.getElementById('btn-en').style.backgroundColor = (lang === 'en') ? '#007bff' : '#6c757d';
  document.getElementById('btn-zh').style.backgroundColor = (lang === 'zh') ? '#007bff' : '#6c757d';
  
  // Load content from external files
  loadContent(lang);
}

// Function to load content from external files
function loadContent(lang) {
  const contentDiv = document.getElementById('content-' + lang);
  
  // Show loading message
  contentDiv.innerHTML = '<div style="text-align: center; padding: 40px; color: #666;"><div class="loading-spinner"></div><p>Loading content...</p></div>';
  
  if (lang === 'en') {
    // Load English content
    fetch('docs/README_en.md')
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.text();
      })
      .then(text => {
        contentDiv.innerHTML = simpleMarkdownParser(text);
      })
      .catch(error => {
        console.error('Error loading English content:', error);
        contentDiv.innerHTML = '<div style="padding: 20px; text-align: center; color: #d32f2f;"><h3>⚠️ Error Loading Content</h3><p>Unable to load English content. Please check the docs/README_en.md file.</p><p>Error: ' + error.message + '</p></div>';
      });
  } else if (lang === 'zh') {
    // Load Chinese content
    fetch('docs/README_zh.md')
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.text();
      })
      .then(text => {
        contentDiv.innerHTML = simpleMarkdownParser(text);
      })
      .catch(error => {
        console.error('Error loading Chinese content:', error);
        contentDiv.innerHTML = '<div style="padding: 20px; text-align: center; color: #d32f2f;"><h3>⚠️ 加载内容时出错</h3><p>无法加载中文内容。请检查 docs/README_zh.md 文件。</p><p>错误: ' + error.message + '</p></div>';
      });
  }
}

// Load English content by default when page loads
window.onload = function() {
  // Add a small delay to ensure DOM is fully loaded
  setTimeout(() => {
    loadContent('en');
  }, 100);
};

// Add error handling for failed script loads
window.addEventListener('error', function(e) {
  console.error('Script error:', e);
}, true);
</script>

<style>
/* Custom styles for better appearance */
button:hover {
  opacity: 0.8;
  transition: opacity 0.2s ease;
}

button:active {
  transform: translateY(1px);
}

#content-en, #content-zh {
  margin-top: 20px;
  min-height: 200px;
}

/* Loading spinner */
.loading-spinner {
  border: 4px solid #f3f3f3;
  border-top: 4px solid #007bff;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Responsive design */
@media (max-width: 768px) {
  button {
    padding: 8px 16px !important;
    margin: 0 5px !important;
    font-size: 14px;
  }
  
  h2 {
    font-size: 1.5em;
  }
}

/* Code block styling */
pre {
  background-color: #f6f8fa;
  border: 1px solid #e1e4e8;
  border-radius: 6px;
  padding: 16px;
  overflow-x: auto;
}

code {
  background-color: #f6f8fa;
  padding: 2px 4px;
  border-radius: 3px;
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}

/* List styling */
li {
  margin: 8px 0;
}

/* Error message styling */
.error-message {
  background-color: #ffebee;
  border: 1px solid #ffcdd2;
  border-radius: 4px;
  padding: 16px;
  margin: 16px 0;
}

/* Button improvements */
button {
  transition: all 0.3s ease;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

button:hover {
  box-shadow: 0 4px 8px rgba(0,0,0,0.2);
  transform: translateY(-1px);
}

button:active {
  transform: translateY(0);
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
