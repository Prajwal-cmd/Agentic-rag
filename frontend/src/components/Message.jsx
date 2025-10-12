import React, { useState } from 'react';
import { User, Bot, ChevronDown, ChevronUp, ExternalLink, FileText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const Message = ({ message }) => {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';
  const isStreaming = message.isStreaming;

  // Custom markdown renderers for proper formatting
  const MarkdownComponents = {
    // Paragraphs with proper spacing
    p: ({ children }) => (
      <p className="mb-3 last:mb-0 leading-7 text-gray-800 dark:text-gray-100">
        {children}
      </p>
    ),
    
    // Bold text - distinct but not overwhelming
    strong: ({ children }) => (
      <strong className="font-semibold text-gray-900 dark:text-white">
        {children}
      </strong>
    ),
    
    // Italic text
    em: ({ children }) => (
      <em className="italic text-gray-700 dark:text-gray-200">
        {children}
      </em>
    ),
    
    // Headings
    h1: ({ children }) => (
      <h1 className="text-2xl font-bold mt-4 mb-3 text-gray-900 dark:text-white">
        {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2 className="text-xl font-bold mt-3 mb-2 text-gray-900 dark:text-white">
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className="text-lg font-semibold mt-2 mb-2 text-gray-900 dark:text-white">
        {children}
      </h3>
    ),
    
    // Lists with proper spacing
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
    li: ({ children }) => (
      <li className="leading-7">
        {children}
      </li>
    ),
    
    // Inline code
    code: ({ inline, children }) => {
      if (inline) {
        return (
          <code className="px-1.5 py-0.5 bg-pink-50 dark:bg-pink-900/30 border border-pink-200 dark:border-pink-700 rounded text-sm font-mono text-pink-700 dark:text-pink-300">
            {children}
          </code>
        );
      }
      // Block code
      return (
        <pre className="bg-gray-900 dark:bg-black p-4 rounded-lg overflow-x-auto my-3 border border-gray-700">
          <code className="text-sm font-mono text-green-400">
            {children}
          </code>
        </pre>
      );
    },
    
    // Links
    a: ({ href, children }) => (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline font-medium"
      >
        {children}
      </a>
    ),
    
    // Blockquotes
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-gray-300 dark:border-gray-600 pl-4 italic my-2 text-gray-700 dark:text-gray-300">
        {children}
      </blockquote>
    ),
  };

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'} mb-4 px-2`}>
      {!isUser && (
        <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 via-blue-600 to-purple-600 flex items-center justify-center shadow-lg ring-2 ring-blue-100 dark:ring-blue-900">
          <Bot className="w-5 h-5 text-white" />
        </div>
      )}

      <div className={`flex flex-col max-w-[80%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`rounded-2xl px-5 py-3 shadow-md ${
            isUser
              ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white'
              : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-50 border border-gray-200 dark:border-gray-700'
          } ${isStreaming ? 'animate-pulse' : ''}`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap break-words leading-7 text-white">
              {message.content}
            </p>
          ) : (
            <div className="prose prose-sm max-w-none dark:prose-invert prose-headings:text-gray-900 dark:prose-headings:text-white prose-p:text-gray-800 dark:prose-p:text-gray-100">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={MarkdownComponents}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* Sources Dropdown - PRESERVED */}
        {!isUser && message.sources && message.sources.length > 0 && !isStreaming && (
          <div className="mt-2 w-full">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors px-3 py-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              <FileText className="w-4 h-4" />
              <span className="font-medium">
                {message.sources.length} {message.sources.length === 1 ? 'Source' : 'Sources'}
              </span>
              {showSources ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>

            {showSources && (
              <div className="mt-2 space-y-2 max-h-96 overflow-y-auto">
                {message.sources.map((source, idx) => (
                  <div
                    key={idx}
                    className="bg-gradient-to-r from-gray-50 to-blue-50 dark:from-gray-800 dark:to-blue-900/20 border border-gray-200 dark:border-gray-700 rounded-lg p-4 text-sm shadow-sm"
                  >
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div className="flex items-center gap-2">
                        <span className="flex-shrink-0 w-7 h-7 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold shadow">
                          {idx + 1}
                        </span>
                        <span className="font-semibold text-gray-900 dark:text-gray-100 break-all">
                          {source.filename || source.title || 'Source'}
                        </span>
                      </div>
                      {source.score && (
                        <span className="flex-shrink-0 text-xs bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 px-2.5 py-1 rounded-full font-semibold border border-green-200 dark:border-green-700">
                          {(source.score * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>

                    <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
                      {source.content}
                    </p>

                    {source.metadata?.page && (
                      <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 font-medium">
                        📄 Page {source.metadata.page}
                      </div>
                    )}

                    {source.url && (
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="mt-2 inline-flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:underline text-xs font-medium"
                      >
                        View source
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Timestamp */}
        <span className="text-xs text-gray-500 dark:text-gray-400 mt-1.5 px-1">
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>

      {isUser && (
        <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center shadow-lg ring-2 ring-emerald-100 dark:ring-emerald-900">
          <User className="w-5 h-5 text-white" />
        </div>
      )}
    </div>
  );
};

export default Message;
