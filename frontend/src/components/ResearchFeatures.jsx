import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { BookOpen, Table2, Calculator, Download, Loader2, AlertCircle, Clock, Search } from 'lucide-react';

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

  // Markdown renderer for synthesis
  const MarkdownComponents = {
    p: ({ children }) => (
      <p className="mb-3 leading-7 text-gray-800 dark:text-gray-100">{children}</p>
    ),
    strong: ({ children }) => (
      <strong className="font-semibold text-gray-900 dark:text-white">{children}</strong>
    ),
    em: ({ children }) => (
      <em className="italic text-gray-700 dark:text-gray-200">{children}</em>
    ),
    ul: ({ children }) => (
      <ul className="list-disc list-inside space-y-1 my-2 ml-2 text-gray-800 dark:text-gray-100">
        {children}
      </ul>
    ),
    ol: ({ children }) => (
      <ol className="list-decimal list-inside space-y-1 my-2 ml-2 text-gray-800 dark:text-gray-100">
        {children}
      </ol>
    ),
    li: ({ children }) => <li className="leading-7">{children}</li>,
    code: ({ inline, children }) => {
      if (inline) {
        return (
          <code className="px-1.5 py-0.5 bg-pink-50 dark:bg-pink-900/30 border border-pink-200 dark:border-pink-700 rounded text-sm font-mono text-pink-700 dark:text-pink-300">
            {children}
          </code>
        );
      }
      return (
        <pre className="bg-gray-900 dark:bg-black p-4 rounded-lg overflow-x-auto my-3 border border-gray-700">
          <code className="text-sm font-mono text-green-400">{children}</code>
        </pre>
      );
    },
    a: ({ href, children }) => (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 dark:text-blue-400 hover:underline font-medium"
      >
        {children}
      </a>
    ),
  };

  const handleLiteratureReview = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const response = await fetch(
        `http://localhost:8000/research/literature-review?research_question=${encodeURIComponent(
          researchQuestion
        )}&max_papers=${maxPapers}&min_year=${minYear}`,
        { method: 'POST' }
      );

      // Handle rate limit (429)
      if (response.status === 429) {
        const data = await response.json();
        setError({
          type: 'rate_limit',
          message: data.detail?.error || 'API rate limit exceeded. Please try again in a few minutes.',
          suggestion: data.detail?.suggestion || 'The system will automatically use arXiv and CORE as fallback sources.',
          retryAfter: data.detail?.retry_after || '5 minutes',
          fallbackAvailable: data.detail?.fallback_available || true,
        });
        return;
      }

      // Handle service unavailable (503)
      if (response.status === 503) {
        const data = await response.json();
        setError({
          type: 'connection_error',
          message: data.detail?.error || 'Unable to connect to research databases.',
          suggestion: data.detail?.suggestion || 'Please check your connection and try again.',
        });
        return;
      }

      // Handle timeout (504)
      if (response.status === 504) {
        const data = await response.json();
        setError({
          type: 'timeout',
          message: data.detail?.error || 'Request timed out.',
          suggestion: data.detail?.suggestion || 'The databases are slow right now. Try again in a moment.',
        });
        return;
      }

      // Handle other errors
      if (!response.ok) {
        const data = await response.json();
        setError({
          type: 'unknown_error',
          message: data.detail?.error || 'Literature review failed',
          suggestion: data.detail?.suggestion || 'Please try again later.',
        });
        return;
      }

      const data = await response.json();

      // Check if no results (from metadata)
      if (data.papers.length === 0 || data.metadata?.error_type === 'no_results') {
        setError({
          type: 'no_results',
          message: data.metadata?.error_message || `No papers found for: "${researchQuestion}"`,
          suggestion: data.metadata?.suggestion || 'Try rephrasing your query or broadening the search terms.',
        });
        return;
      }

      setResult(data);
    } catch (err) {
      setError({
        type: 'unknown_error',
        message: err.message || 'Failed to conduct literature review',
        suggestion: 'Please try again later or check your connection.',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleTableExtraction = async () => {
    if (!pdfFile) {
      setError({ type: 'validation', message: 'Please select a PDF file', suggestion: '' });
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', pdfFile);
      const response = await fetch(
        `http://localhost:8000/research/extract-tables?pages=${pages}&output_format=${tableFormat}`,
        { method: 'POST', body: formData }
      );
      if (!response.ok) throw new Error('Table extraction failed');
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError({ type: 'unknown_error', message: err.message, suggestion: 'Try again later' });
    } finally {
      setLoading(false);
    }
  };

  const handleMathExtraction = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `http://localhost:8000/research/extract-math?text=${encodeURIComponent(
          mathText
        )}&verify_latex=true&fix_errors=true`,
        { method: 'POST' }
      );
      if (!response.ok) throw new Error('Math extraction failed');
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError({ type: 'unknown_error', message: err.message, suggestion: 'Try again later' });
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
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-2xl p-6 shadow-xl">
        <h1 className="text-3xl font-bold mb-2">Research Tools</h1>
        <p className="text-purple-100">
          Advanced features for literature review, table extraction, and mathematical analysis
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-200 dark:border-gray-700">
        <button
          onClick={() => {
            setActiveTab('literature');
            setResult(null);
            setError(null);
          }}
          className={`flex items-center gap-2 px-4 py-3 font-medium border-b-2 transition-colors ${
            activeTab === 'literature'
              ? 'border-purple-600 text-purple-600 dark:text-purple-400'
              : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
          }`}
        >
          <BookOpen className="w-5 h-5" />
          Literature Review
        </button>
        <button
          onClick={() => {
            setActiveTab('tables');
            setResult(null);
            setError(null);
          }}
          className={`flex items-center gap-2 px-4 py-3 font-medium border-b-2 transition-colors ${
            activeTab === 'tables'
              ? 'border-purple-600 text-purple-600 dark:text-purple-400'
              : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
          }`}
        >
          <Table2 className="w-5 h-5" />
          Table Extraction
        </button>
        <button
          onClick={() => {
            setActiveTab('math');
            setResult(null);
            setError(null);
          }}
          className={`flex items-center gap-2 px-4 py-3 font-medium border-b-2 transition-colors ${
            activeTab === 'math'
              ? 'border-purple-600 text-purple-600 dark:text-purple-400'
              : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
          }`}
        >
          <Calculator className="w-5 h-5" />
          Math Extraction
        </button>
      </div>

      {/* Enhanced Error Display with Different Types */}
      {error && (
        <div
          className={`rounded-lg border p-4 flex items-start gap-3 ${
            error.type === 'rate_limit'
              ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
              : error.type === 'no_results'
              ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
              : 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
          }`}
        >
          <div className="flex-shrink-0 mt-0.5">
            {error.type === 'rate_limit' ? (
              <Clock className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
            ) : error.type === 'no_results' ? (
              <Search className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            ) : (
              <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
            )}
          </div>
          <div className="flex-1">
            <h3
              className={`font-semibold mb-1 ${
                error.type === 'rate_limit'
                  ? 'text-yellow-900 dark:text-yellow-100'
                  : error.type === 'no_results'
                  ? 'text-blue-900 dark:text-blue-100'
                  : 'text-red-900 dark:text-red-100'
              }`}
            >
              {error.type === 'rate_limit' && 'Rate Limit Exceeded'}
              {error.type === 'no_results' && 'No Results Found'}
              {error.type === 'connection_error' && 'Connection Error'}
              {error.type === 'timeout' && 'Request Timeout'}
              {error.type === 'unknown_error' && 'Error'}
              {error.type === 'validation' && 'Validation Error'}
            </h3>
            <p
              className={`text-sm mb-2 ${
                error.type === 'rate_limit'
                  ? 'text-yellow-700 dark:text-yellow-300'
                  : error.type === 'no_results'
                  ? 'text-blue-700 dark:text-blue-300'
                  : 'text-red-700 dark:text-red-300'
              }`}
            >
              {error.message}
            </p>
            {error.suggestion && (
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                💡 {error.suggestion}
              </p>
            )}
            {error.retryAfter && error.type === 'rate_limit' && (
              <p className="text-xs text-yellow-700 dark:text-yellow-300 mt-2 font-medium">
                ⏱️ Retry after: {error.retryAfter}
              </p>
            )}
            {error.type === 'rate_limit' && error.fallbackAvailable && (
              <div className="mt-3 text-xs text-yellow-700 dark:text-yellow-300 bg-yellow-100 dark:bg-yellow-900/40 rounded px-2 py-1.5">
                <strong>Note:</strong> The system automatically tries arXiv and CORE when Semantic
                Scholar rate limits are reached. Most queries will still work with these fallback
                sources.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Literature Review Tab */}
      {activeTab === 'literature' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-lg">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              Conduct Literature Review
            </h2>
            <p className="text-gray-600 dark:text-gray-300 mb-6">
              Conduct Elicit-style literature reviews with multi-paper synthesis
            </p>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Research Question
                </label>
                <input
                  type="text"
                  value={researchQuestion}
                  onChange={(e) => setResearchQuestion(e.target.value)}
                  placeholder="What are the effects of climate change on biodiversity?"
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 dark:focus:ring-purple-400 outline-none bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Max Papers
                  </label>
                  <input
                    type="number"
                    value={maxPapers}
                    onChange={(e) => setMaxPapers(parseInt(e.target.value))}
                    min="1"
                    max="50"
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 dark:focus:ring-purple-400 outline-none bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Min Year
                  </label>
                  <input
                    type="number"
                    value={minYear}
                    onChange={(e) => setMinYear(parseInt(e.target.value))}
                    min="1900"
                    max={new Date().getFullYear()}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 dark:focus:ring-purple-400 outline-none bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
              </div>

              <button
                onClick={handleLiteratureReview}
                disabled={!researchQuestion.trim() || loading}
                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-gray-400 disabled:to-gray-400 text-white py-3 px-6 rounded-lg font-semibold transition-all disabled:cursor-not-allowed shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing Papers...
                  </>
                ) : (
                  'Start Review'
                )}
              </button>
            </div>
          </div>

          {/* Results Display with Markdown Rendering */}
          {result && (
            <div className="space-y-6">
              {/* Synthesis Section with Markdown */}
              {result.synthesis && (
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl border border-blue-200 dark:border-blue-800 p-6 shadow-lg">
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                    📊 Research Synthesis
                  </h3>
                  <div className="prose prose-sm max-w-none dark:prose-invert">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={MarkdownComponents}>
                      {result.synthesis}
                    </ReactMarkdown>
                  </div>
                </div>
              )}

              {/* Citation Export */}
              {result.citation_export && (
                <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-lg">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    📚 Export Citations
                  </h3>
                  <div className="flex gap-3">
                    {['bibtex', 'ris'].map((format) => (
                      <button
                        key={format}
                        onClick={() => downloadCitation(format)}
                        className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
                      >
                        <Download className="w-4 h-4" />
                        {format.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Papers List */}
              {result.papers && result.papers.length > 0 && (
                <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-lg">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    📖 Papers Analyzed ({result.papers.length})
                  </h3>
                  <div className="space-y-4">
                    {result.papers.map((paper, idx) => (
                      <div
                        key={idx}
                        className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow"
                      >
                        {/* Paper Title with Source Badge */}
                        <div className="flex items-start justify-between gap-3 mb-2">
                          <h4 className="font-semibold text-gray-900 dark:text-white flex-1">
                            {paper.title}
                          </h4>
                          {paper.source && (
                            <span
                              className={`text-xs px-2 py-1 rounded-full font-semibold flex-shrink-0 ${
                                paper.source === 'semantic_scholar'
                                  ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                                  : paper.source === 'arxiv'
                                  ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300'
                                  : 'bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300'
                              }`}
                            >
                              {paper.source === 'semantic_scholar'
                                ? 'Semantic Scholar'
                                : paper.source === 'arxiv'
                                ? 'arXiv'
                                : 'CORE'}
                            </span>
                          )}
                        </div>

                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                          {paper.authors?.slice(0, 3).join(', ')}
                          {paper.authors?.length > 3 && ' et al.'} ({paper.year}) · {paper.citations}{' '}
                          citations
                        </p>

                        {paper.key_findings && paper.key_findings.length > 0 && (
                          <div className="mt-2">
                            <p className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                              Key Findings:
                            </p>
                            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1 ml-4 list-disc">
                              {paper.key_findings.map((finding, fidx) => (
                                <li key={fidx}>{finding}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {paper.url && (
                          <a
                            href={paper.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-block mt-2 text-xs text-blue-600 dark:text-blue-400 hover:underline"
                          >
                            View Paper →
                          </a>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Table Extraction Tab */}
      {activeTab === 'tables' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-lg">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              Extract Tables from PDF
            </h2>
            <p className="text-gray-600 dark:text-gray-300 mb-6">
              Extract tables from research papers and export to CSV/Excel
            </p>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  PDF File
                </label>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={(e) => setPdfFile(e.target.files[0])}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 dark:focus:ring-purple-400 outline-none bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Pages
                  </label>
                  <input
                    type="text"
                    value={pages}
                    onChange={(e) => setPages(e.target.value)}
                    placeholder="all or 1-5"
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 dark:focus:ring-purple-400 outline-none bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Format
                  </label>
                  <select
                    value={tableFormat}
                    onChange={(e) => setTableFormat(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 dark:focus:ring-purple-400 outline-none bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  >
                    <option value="csv">CSV</option>
                    <option value="json">JSON</option>
                    <option value="markdown">Markdown</option>
                  </select>
                </div>
              </div>

              <button
                onClick={handleTableExtraction}
                disabled={!pdfFile || loading}
                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-gray-400 disabled:to-gray-400 text-white py-3 px-6 rounded-lg font-semibold transition-all disabled:cursor-not-allowed shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Extracting Tables...
                  </>
                ) : (
                  'Extract Tables'
                )}
              </button>
            </div>
          </div>

          {result && result.tables && (
            <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-lg">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                Extracted Tables ({result.tables.length})
              </h3>
              <div className="space-y-4">
                {result.tables.map((table, idx) => (
                  <div
                    key={idx}
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
                  >
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                      {table.filename}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Page {table.page}, {table.rows} rows × {table.columns} cols
                    </p>
                    <pre className="mt-2 bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
                      {table.content}
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Math Extraction Tab */}
      {activeTab === 'math' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-lg">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              Extract Mathematical Formulas
            </h2>
            <p className="text-gray-600 dark:text-gray-300 mb-6">
              Extract LaTeX formulas from text with automatic rendering
            </p>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Text with Math
                </label>
                <textarea
                  value={mathText}
                  onChange={(e) => setMathText(e.target.value)}
                  placeholder="The equation is $E = mc^2$ and..."
                  rows={6}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 dark:focus:ring-purple-400 outline-none resize-none bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 font-mono text-sm"
                />
              </div>

              <button
                onClick={handleMathExtraction}
                disabled={!mathText.trim() || loading}
                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-gray-400 disabled:to-gray-400 text-white py-3 px-6 rounded-lg font-semibold transition-all disabled:cursor-not-allowed shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Extracting Math...
                  </>
                ) : (
                  'Extract Formulas'
                )}
              </button>
            </div>
          </div>

          {result && result.formulas && (
            <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-lg">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                Extracted Formulas ({result.inline_count} inline, {result.block_count} block)
              </h3>
              <div className="space-y-4">
                {result.formulas.map((formula, idx) => (
                  <div
                    key={idx}
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <pre className="bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm mb-2 overflow-x-auto">
                          {formula.latex}
                        </pre>
                        {formula.valid === false && (
                          <span className="text-xs text-red-600 dark:text-red-400 font-medium">
                            ⚠️ Invalid LaTeX
                          </span>
                        )}
                        {formula.fixed_latex && (
                          <span className="text-xs text-green-600 dark:text-green-400 font-medium ml-2">
                            ✓ Auto-fixed
                          </span>
                        )}
                      </div>
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
