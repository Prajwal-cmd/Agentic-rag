import React, { useState } from 'react';
import { User, Bot, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import { formatTimestamp } from '../utils/formatters';

const Message = ({ message }) => {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';
  const isStreaming = message.isStreaming;

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
      <div className={`flex ${isUser ? 'flex-row-reverse' : 'flex-row'} max-w-4xl w-full`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 ${isUser ? 'ml-3' : 'mr-3'}`}>
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
            isUser ? 'bg-blue-500' : 'bg-gray-700'
          }`}>
            {isUser ? (
              <User size={18} className="text-white" />
            ) : (
              <Bot size={18} className="text-white" />
            )}
          </div>
        </div>
        
        {/* Message Content */}
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} flex-1 min-w-0`}>
          <div className={`px-4 py-3 rounded-lg ${
            isUser 
              ? 'bg-blue-500 text-white' 
              : 'bg-gray-100 text-gray-900 border border-gray-200'
          } max-w-full`}>
            <div className="whitespace-pre-wrap break-words">
              {message.content}
              {isStreaming && (
                <span className="inline-block w-2 h-4 bg-gray-400 ml-1 animate-pulse" />
              )}
            </div>
          </div>
          
          {/* Timestamp */}
          <span className="text-xs text-gray-500 mt-1 px-1">
            {formatTimestamp(message.timestamp)}
          </span>

          {/* Sources */}
          {!isUser && message.sources && message.sources.length > 0 && (
            <div className="mt-2 w-full">
              <button
                onClick={() => setShowSources(!showSources)}
                className="flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900 transition-colors"
              >
                {showSources ? (
                  <ChevronUp size={16} />
                ) : (
                  <ChevronDown size={16} />
                )}
                <span>{message.sources.length} source{message.sources.length > 1 ? 's' : ''}</span>
              </button>
              
              {showSources && (
                <div className="mt-2 space-y-2">
                  {message.sources.map((source, idx) => (
                    <div 
                      key={idx} 
                      className="text-sm bg-white p-3 rounded-lg border border-gray-200 shadow-sm"
                    >
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <div className="font-medium text-gray-900 flex-1">
                          {source.title}
                        </div>
                        {source.type && (
                          <span className={`text-xs px-2 py-0.5 rounded ${
                            source.type === 'web_search' 
                              ? 'bg-green-100 text-green-700' 
                              : 'bg-blue-100 text-blue-700'
                          }`}>
                            {source.type === 'web_search' ? 'Web' : 'Doc'}
                          </span>
                        )}
                      </div>
                      
                      {source.url && (
                        <a 
                          href={source.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline text-xs flex items-center gap-1 mb-2"
                        >
                          <ExternalLink size={12} />
                          {source.url.length > 60 ? source.url.substring(0, 60) + '...' : source.url}
                        </a>
                      )}
                      
                      <p className="text-gray-600 text-xs leading-relaxed">
                        {source.content}
                      </p>
                      
                      {source.score && (
                        <div className="text-xs text-gray-500 mt-2">
                          Relevance: {(source.score * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message;