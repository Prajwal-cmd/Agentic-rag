import React, { createContext, useContext, useState, useEffect } from 'react';

const ChatContext = createContext();

export const useChatContext = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChatContext must be used within ChatProvider');
  }
  return context;
};

export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [documentsUploaded, setDocumentsUploaded] = useState(false);
  const [uploadedDocuments, setUploadedDocuments] = useState([]); // NEW: Track uploaded documents
  const [error, setError] = useState(null);

  // Generate session ID on mount
  useEffect(() => {
    const generateSessionId = () => {
      const timestamp = Date.now();
      const random1 = Math.random().toString(36).substr(2, 9);
      const random2 = crypto.randomUUID ? crypto.randomUUID().split('-')[0] : Math.random().toString(36).substr(2, 9);
      return `${timestamp}-${random1}-${random2}`;
    };

    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    console.log('Session ID generated:', newSessionId);
  }, []);

  const addMessage = (role, content, sources = null) => {
    const newMessage = {
      id: `${Date.now()}-${Math.random()}`,
      role,
      content,
      sources: sources || [],
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, newMessage]);
    return newMessage.id;
  };

  const updateLastMessage = (content, sources = null) => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const updated = [...prev];
      const lastMessage = updated[updated.length - 1];
      updated[updated.length - 1] = {
        ...lastMessage,
        content: content,
        sources: sources || lastMessage.sources || [],
      };
      return updated;
    });
  };

  const addStreamingMessage = (role, initialContent = '') => {
    const messageId = `${Date.now()}-${Math.random()}`;
    const newMessage = {
      id: messageId,
      role,
      content: initialContent,
      sources: [],
      timestamp: new Date().toISOString(),
      isStreaming: true,
    };
    setMessages((prev) => [...prev, newMessage]);
    return messageId;
  };

  const appendToStreamingMessage = (messageId, token) => {
    setMessages((prev) => {
      return prev.map((msg) => {
        if (msg.id === messageId) {
          return {
            ...msg,
            content: msg.content + token,
          };
        }
        return msg;
      });
    });
  };

  const finalizeStreamingMessage = (messageId, sources = []) => {
    setMessages((prev) => {
      return prev.map((msg) => {
        if (msg.id === messageId) {
          return {
            ...msg,
            isStreaming: false,
            sources: sources,
          };
        }
        return msg;
      });
    });
  };

  const clearMessages = () => {
    setMessages([]);
  };

  // NEW: Add uploaded document to list
  const addUploadedDocument = (filename, chunksCount) => {
    setUploadedDocuments((prev) => [
      ...prev,
      {
        id: `${Date.now()}-${Math.random()}`,
        filename,
        chunksCount,
        timestamp: new Date().toISOString(),
      },
    ]);
  };

  const value = {
    messages,
    sessionId,
    isLoading,
    documentsUploaded,
    uploadedDocuments, // NEW
    error,
    setIsLoading,
    setDocumentsUploaded,
    setError,
    addMessage,
    updateLastMessage,
    addStreamingMessage,
    appendToStreamingMessage,
    finalizeStreamingMessage,
    clearMessages,
    addUploadedDocument, // NEW
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
};
