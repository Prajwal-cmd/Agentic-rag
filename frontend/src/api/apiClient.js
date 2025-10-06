import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ========== EXISTING CHAT API ==========
export const chatAPI = {
  sendMessage: async (message, conversationHistory, sessionId) => {
    const response = await apiClient.post('/chat', {
      message,
      conversation_history: conversationHistory,
      session_id: sessionId,
    });
    return response.data;
  },

  // Streaming endpoint
  sendMessageStream: async (message, conversationHistory, sessionId, callbacks) => {
    const { onProgress, onToken, onComplete, onError } = callbacks;

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          conversation_history: conversationHistory,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          const eventMatch = line.match(/^event: (.+)$/m);
          const dataMatch = line.match(/^data: (.+)$/m);

          if (eventMatch && dataMatch) {
            const eventType = eventMatch[1];
            const data = JSON.parse(dataMatch[1]);

            switch (eventType) {
              case 'progress':
                onProgress?.(data);
                break;
              case 'token':
                onToken?.(data.token);
                break;
              case 'complete':
                onComplete?.(data);
                break;
              case 'error':
                onError?.(data.message);
                break;
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      onError?.(error.message);
      throw error;
    }
  },

  uploadDocuments: async (files, sessionId) => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    const response = await apiClient.post(`/upload?session_id=${sessionId}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  deleteSession: async (sessionId) => {
    const response = await apiClient.delete(`/session/${sessionId}`);
    return response.data;
  },

  healthCheck: async () => {
    const response = await apiClient.get('/health');
    return response.data;
  },
};

// ========== NEW: RESEARCH API ==========
export const researchAPI = {
  /**
   * Conduct automated literature review (Elicit-style)
   * @param {string} researchQuestion - Research question to investigate
   * @param {number} maxPapers - Maximum papers to analyze (1-20)
   * @param {number} minYear - Minimum publication year
   * @param {boolean} extractStructured - Extract key findings, methods, limitations
   * @returns {Promise} Literature review results
   */
  conductLiteratureReview: async (researchQuestion, maxPapers = 10, minYear = 2020, extractStructured = true) => {
    try {
      const response = await apiClient.post(
        `/research/literature-review`,
        null,
        {
          params: {
            research_question: researchQuestion,
            max_papers: maxPapers,
            min_year: minYear,
            extract_structured: extractStructured
          }
        }
      );
      return response.data;
    } catch (error) {
      console.error('Literature review error:', error);
      throw error;
    }
  },

  /**
   * Export citations in BibTeX or RIS format
   * @param {Array<string>} paperIds - Paper IDs to export
   * @param {string} format - 'bibtex' or 'ris'
   * @returns {Promise} Citation export
   */
  exportCitations: async (paperIds, format = 'bibtex') => {
    try {
      const response = await apiClient.post(
        `/research/export-citations`,
        null,
        {
          params: {
            paper_ids: paperIds,
            format: format
          }
        }
      );
      return response.data;
    } catch (error) {
      console.error('Citation export error:', error);
      throw error;
    }
  },

  /**
   * Extract tables from PDF
   * @param {File} file - PDF file
   * @param {string} pages - Pages to extract ('all', '1-3', '1,3,5')
   * @param {string} outputFormat - 'csv', 'excel', or 'markdown'
   * @returns {Promise} Extracted tables
   */
  extractTables: async (file, pages = 'all', outputFormat = 'csv') => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await apiClient.post(
        `/research/extract-tables`,
        formData,
        {
          params: {
            pages: pages,
            output_format: outputFormat
          },
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('Table extraction error:', error);
      throw error;
    }
  },

  /**
   * Extract and verify mathematical formulas from text
   * @param {string} text - Text containing math (LaTeX or Unicode)
   * @param {boolean} verifyLatex - Verify LaTeX syntax
   * @param {boolean} fixErrors - Attempt to fix broken LaTeX with LLM
   * @returns {Promise} Extracted formulas
   */
  extractMath: async (text, verifyLatex = true, fixErrors = true) => {
    try {
      const response = await apiClient.post(
        `/research/extract-math`,
        null,
        {
          params: {
            text: text,
            verify_latex: verifyLatex,
            fix_errors: fixErrors
          }
        }
      );
      return response.data;
    } catch (error) {
      console.error('Math extraction error:', error);
      throw error;
    }
  },

  /**
   * Download citation file (helper function)
   * @param {string} content - Citation content (BibTeX or RIS)
   * @param {string} filename - Output filename
   */
  downloadCitation: (content, filename) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },

  /**
   * Download table file (helper function)
   * @param {string} content - Table content
   * @param {string} filename - Output filename
   * @param {string} type - MIME type
   */
  downloadTable: (content, filename, type = 'text/csv') => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
};

export default apiClient;
