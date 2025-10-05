import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

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

export default apiClient;