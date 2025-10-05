import React, { useState } from 'react';
import { useChatContext } from '../context/ChatContext';
import { chatAPI } from '../api/apiClient';
import MessageList from './MessageList';
import InputBox from './InputBox';
import DocumentUpload from './DocumentUpload';
import { AlertCircle, X } from 'lucide-react';

const ChatInterface = () => {
  const { 
    messages, 
    sessionId, 
    isLoading, 
    setIsLoading, 
    error,
    setError, 
    addMessage,
    addStreamingMessage,
    appendToStreamingMessage,
    finalizeStreamingMessage
  } = useChatContext();

  const [currentProgress, setCurrentProgress] = useState(null);
  const [useStreaming] = useState(true); // Set to false to use non-streaming

  const handleSendMessage = async (message) => {
    if (!sessionId) {
      setError('Session not initialized. Please refresh the page.');
      return;
    }

    console.log('Sending message with session ID:', sessionId);

    // Add user message
    addMessage('user', message);
    setIsLoading(true);
    setError(null);
    setCurrentProgress(null);

    try {
      // Convert messages to API format
      const conversationHistory = messages.map(msg => ({
        role: msg.role,
        content: msg.content,
      }));

      if (useStreaming) {
        // Use streaming endpoint
        const assistantMessageId = addStreamingMessage('assistant', '');

        await chatAPI.sendMessageStream(
          message, 
          conversationHistory, 
          sessionId,
          {
            onProgress: (data) => {
              setCurrentProgress(data.message);
              console.log('Progress:', data.message);
            },
            onToken: (token) => {
              appendToStreamingMessage(assistantMessageId, token);
            },
            onComplete: (data) => {
              console.log('Complete:', data);
              finalizeStreamingMessage(assistantMessageId, data.sources || []);
              setCurrentProgress(null);
              setIsLoading(false);
            },
            onError: (errorMsg) => {
              console.error('Stream error:', errorMsg);
              setError(errorMsg);
              finalizeStreamingMessage(assistantMessageId, []);
              setIsLoading(false);
            }
          }
        );
      } else {
        // Use non-streaming endpoint
        const response = await chatAPI.sendMessage(message, conversationHistory, sessionId);
        
        console.log('Chat response:', response);
        console.log('Route taken:', response.route_taken);
        
        // Add assistant message with sources
        addMessage('assistant', response.answer, response.sources);
        setIsLoading(false);
      }
      
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to get response';
      console.error('Send message error:', error);
      setError(errorMsg);
      addMessage('assistant', `Error: ${errorMsg}`);
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Left Panel - Document Upload */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <h1 className="text-xl font-semibold text-gray-900">Agentic RAG</h1>
          {sessionId && (
            <p className="text-xs text-gray-500 mt-1">
              Session: {sessionId.substring(0, 15)}...
            </p>
          )}
        </div>
        <div className="flex-1 overflow-y-auto">
          <DocumentUpload />
        </div>
      </div>

      {/* Right Panel - Chat */}
      <div className="flex-1 flex flex-col">
        {/* Error Banner */}
        {error && (
          <div className="bg-red-50 border-b border-red-200 p-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-red-700">
              <AlertCircle size={18} />
              <span className="text-sm">{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-red-500 hover:text-red-700"
            >
              <X size={18} />
            </button>
          </div>
        )}

        {/* Progress Indicator */}
        {currentProgress && (
          <div className="bg-blue-50 border-b border-blue-200 p-3">
            <div className="flex items-center gap-2 text-blue-700">
              <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm">{currentProgress}</span>
            </div>
          </div>
        )}

        {/* Message List */}
        <MessageList />

        {/* Input Box */}
        <InputBox onSendMessage={handleSendMessage} />
      </div>
    </div>
  );
};

export default ChatInterface;