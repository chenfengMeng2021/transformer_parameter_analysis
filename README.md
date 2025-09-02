# AI Block Research

<div align="center">
  <h2>Transformer Parameter Analysis</h2>
  <p>A research toolkit for downloading, reading, and analyzing large language model parameters</p>
  
  <div style="margin: 20px 0;">
    <button onclick="showLanguage('en')" id="btn-en" style="padding: 10px 20px; margin: 0 10px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">English</button>
    <button onclick="showLanguage('zh')" id="btn-zh" style="padding: 10px 20px; margin: 0 10px; background-color: #6c757d; color: white; border: none; border-radius: 5px; cursor: pointer;">中文</button>
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
  
  if (lang === 'en') {
    // Load English content
    fetch('./docs/README_en.md')
      .then(response => response.text())
      .then(text => {
        contentDiv.innerHTML = marked.parse(text);
      })
      .catch(error => {
        contentDiv.innerHTML = '<p>Error loading English content. Please check the docs/README_en.md file.</p>';
      });
  } else if (lang === 'zh') {
    // Load Chinese content
    fetch('./docs/README_zh.md')
      .then(response => response.text())
      .then(text => {
        contentDiv.innerHTML = marked.parse(text);
      })
      .catch(error => {
        contentDiv.innerHTML = '<p>加载中文内容时出错。请检查 docs/README_zh.md 文件。</p>';
      });
  }
}

// Load English content by default when page loads
window.onload = function() {
  loadContent('en');
};
</script>

<!-- Include marked.js for Markdown parsing -->
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

<style>
/* Custom styles for better appearance */
button:hover {
  opacity: 0.8;
}

button:active {
  transform: translateY(1px);
}

#content-en, #content-zh {
  margin-top: 20px;
}

/* Responsive design */
@media (max-width: 768px) {
  button {
    padding: 8px 16px !important;
    margin: 0 5px !important;
    font-size: 14px;
  }
}
</style>
