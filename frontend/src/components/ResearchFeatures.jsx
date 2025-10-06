import React, { useState } from 'react';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const ResearchFeatures = ({ sessionId }) => {
  const [activeTab, setActiveTab] = useState('literature');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Literature Review State
  const [researchQuestion, setResearchQuestion] = useState('');
  const [maxPapers, setMaxPapers] = useState(10);
  const [minYear, setMinYear] = useState(2020);

  // Table Extraction State
  const [pdfFile, setPdfFile] = useState(null);
  const [pages, setPages] = useState('all');
  const [tableFormat, setTableFormat] = useState('csv');

  // Math Extraction State
  const [mathText, setMathText] = useState('');

  const handleLiteratureReview = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(
        `http://localhost:8000/research/literature-review?research_question=${encodeURIComponent(researchQuestion)}&max_papers=${maxPapers}&min_year=${minYear}`,
        { method: 'POST' }
      );
      
      if (!response.ok) throw new Error('Literature review failed');
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTableExtraction = async () => {
    if (!pdfFile) {
      setError('Please select a PDF file');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', pdfFile);
      
      const response = await fetch(
        `http://localhost:8000/research/extract-tables?pages=${pages}&output_format=${tableFormat}`,
        {
          method: 'POST',
          body: formData
        }
      );
      
      if (!response.ok) throw new Error('Table extraction failed');
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleMathExtraction = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(
        `http://localhost:8000/research/extract-math?text=${encodeURIComponent(mathText)}&verify_latex=true&fix_errors=true`,
        { method: 'POST' }
      );
      
      if (!response.ok) throw new Error('Math extraction failed');
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadCitation = (format) => {
    if (!result?.citation_export?.[format]) return;
    
    const blob = new Blob([result.citation_export[format]], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `citations.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Tabs */}
      <div className="border-b border-gray-200 mb-8">
        <nav className="flex space-x-8">
          <button
            onClick={() => setActiveTab('literature')}
            className={`pb-4 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'literature'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            📚 Literature Review
          </button>
          <button
            onClick={() => setActiveTab('tables')}
            className={`pb-4 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'tables'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            📊 Table Extraction
          </button>
          <button
            onClick={() => setActiveTab('math')}
            className={`pb-4 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'math'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            ∑ Math Formulas
          </button>
        </nav>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
          {error}
        </div>
      )}

      {/* Literature Review Tab */}
      {activeTab === 'literature' && (
        <div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-2">Automated Literature Review</h2>
          <p className="text-gray-600 mb-6">
            Conduct Elicit-style literature reviews with multi-paper synthesis
          </p>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Research Question
              </label>
              <textarea
                value={researchQuestion}
                onChange={(e) => setResearchQuestion(e.target.value)}
                placeholder="E.g., What are the latest advances in transformer architectures?"
                rows={3}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Papers
                </label>
                <input
                  type="number"
                  value={maxPapers}
                  onChange={(e) => setMaxPapers(parseInt(e.target.value))}
                  min={1}
                  max={20}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Min Year
                </label>
                <input
                  type="number"
                  value={minYear}
                  onChange={(e) => setMinYear(parseInt(e.target.value))}
                  min={1900}
                  max={new Date().getFullYear()}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>

            <button
              onClick={handleLiteratureReview}
              disabled={loading || !researchQuestion}
              className="px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Analyzing Papers...' : 'Start Literature Review'}
            </button>
          </div>

          {/* Results */}
          {result && result.synthesis && (
            <div className="mt-8 space-y-8">
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Synthesis</h3>
                <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 whitespace-pre-wrap">
                  {result.synthesis}
                </div>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">
                  Papers Analyzed ({result.papers?.length || 0})
                </h3>
                <div className="space-y-4">
                  {result.papers?.map((paper, i) => (
                    <div key={i} className="bg-white border border-gray-200 rounded-lg p-6 hover:border-blue-500 hover:shadow-md transition-all">
                      <h4 className="text-lg font-semibold text-gray-900 mb-2">{paper.title}</h4>
                      <p className="text-sm text-gray-600 mb-3">
                        {paper.authors?.slice(0, 3).join(', ')} et al. ({paper.year}) · {paper.citations} citations
                      </p>
                      {paper.key_findings?.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-100">
                          <p className="text-sm font-medium text-gray-700 mb-2">Key Findings:</p>
                          <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                            {paper.key_findings.map((finding, j) => (
                              <li key={j}>{finding}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      <a
                        href={paper.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-block mt-3 text-blue-600 hover:text-blue-800 text-sm font-medium"
                      >
                        View Paper →
                      </a>
                    </div>
                  ))}
                </div>
              </div>

              {result.key_themes?.length > 0 && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-4">Key Themes</h3>
                  <div className="flex flex-wrap gap-2">
                    {result.key_themes.map((theme, i) => (
                      <span
                        key={i}
                        className="px-4 py-2 bg-blue-50 text-blue-700 border border-blue-200 rounded-full text-sm font-medium"
                      >
                        {theme}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Export Citations</h3>
                <div className="flex gap-3">
                  <button
                    onClick={() => downloadCitation('bibtex')}
                    className="px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 hover:border-blue-500 transition-colors"
                  >
                    Download BibTeX
                  </button>
                  <button
                    onClick={() => downloadCitation('ris')}
                    className="px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 hover:border-blue-500 transition-colors"
                  >
                    Download RIS
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Table Extraction Tab */}
      {activeTab === 'tables' && (
        <div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-2">PDF Table Extraction</h2>
          <p className="text-gray-600 mb-6">
            Extract tables from research papers and export to CSV/Excel
          </p>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload PDF
              </label>
              <input
                type="file"
                accept=".pdf"
                onChange={(e) => setPdfFile(e.target.files[0])}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Pages
                </label>
                <input
                  type="text"
                  value={pages}
                  onChange={(e) => setPages(e.target.value)}
                  placeholder="all, 1-3, or 1,3,5"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Format
                </label>
                <select
                  value={tableFormat}
                  onChange={(e) => setTableFormat(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="csv">CSV</option>
                  <option value="excel">Excel</option>
                  <option value="markdown">Markdown</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleTableExtraction}
              disabled={loading || !pdfFile}
              className="px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Extracting...' : 'Extract Tables'}
            </button>
          </div>

          {result && result.format === 'markdown' && (
            <div className="mt-8">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Extracted Tables (Markdown)</h3>
              <pre className="bg-gray-50 border border-gray-200 rounded-lg p-6 overflow-x-auto text-sm whitespace-pre-wrap max-h-96 overflow-y-auto">
                {result.content}
              </pre>
            </div>
          )}

          {result && result.files && (
            <div className="mt-8">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Extracted {result.files.length} Tables
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {result.files.map((file, i) => (
                  <div key={i} className="bg-white border border-gray-200 rounded-lg p-4 hover:border-blue-500 hover:shadow-md transition-all">
                    <p className="font-semibold text-gray-900">{file.filename}</p>
                    <p className="text-sm text-gray-600 mt-2">
                      Page {file.page}, {file.rows} rows × {file.columns} cols
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Math Extraction Tab */}
      {activeTab === 'math' && (
        <div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-2">Mathematical Formula Extraction</h2>
          <p className="text-gray-600 mb-6">
            Extract LaTeX formulas from text with automatic rendering
          </p>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Text with Math
              </label>
              <textarea
                value={mathText}
                onChange={(e) => setMathText(e.target.value)}
                placeholder="Paste text containing math (LaTeX or Unicode)..."
                rows={6}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
              />
            </div>

            <button
              onClick={handleMathExtraction}
              disabled={loading || !mathText}
              className="px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Extracting...' : 'Extract & Render Math'}
            </button>
          </div>

          {result && result.formulas && (
            <div className="mt-8">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Found {result.total_count} Formulas
              </h3>
              <p className="text-gray-600 mb-4">
                {result.inline_count} inline, {result.block_count} block
              </p>

              <div className="space-y-6">
                {result.formulas.map((formula, i) => (
                  <div key={i} className="bg-white border border-gray-200 rounded-lg p-6 hover:border-blue-500 transition-all">
                    <div className="bg-gray-50 rounded-lg p-6 mb-4 text-center overflow-x-auto">
                      {formula.type === 'inline' ? (
                        <InlineMath math={formula.render_latex} />
                      ) : (
                        <BlockMath math={formula.render_latex} />
                      )}
                    </div>
                    <div className="flex items-center gap-3">
                      <code className="flex-1 bg-gray-100 px-3 py-2 rounded text-sm font-mono text-gray-700">
                        {formula.latex}
                      </code>
                      {formula.valid === false && (
                        <span className="px-3 py-1 bg-red-50 text-red-700 rounded text-xs font-medium">
                          ⚠️ Invalid LaTeX
                        </span>
                      )}
                      {formula.fixed_latex && (
                        <span className="px-3 py-1 bg-green-50 text-green-700 rounded text-xs font-medium">
                          ✓ Auto-fixed
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ResearchFeatures;
