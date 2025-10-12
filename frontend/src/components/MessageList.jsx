import React, { useEffect, useRef } from 'react';
import Message from './Message';
import { useChatContext } from '../context/ChatContext';
import { MessageSquare, Sparkles } from 'lucide-react';

const MessageList = () => {
  const { messages } = useChatContext();
  const messagesEndRef = useRef(null);
  const containerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Filter out system messages for display
  const displayMessages = messages.filter((msg) => msg.role !== 'system');

  if (displayMessages.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-400 dark:text-gray-500 p-8">
        <div className="relative">
          <MessageSquare className="w-20 h-20 mb-4 opacity-50" />
          <Sparkles className="w-8 h-8 absolute -top-2 -right-2 text-blue-500 animate-pulse" />
        </div>
        <h3 className="text-2xl font-bold mb-2 text-gray-700 dark:text-gray-300">
          Start Your Conversation
        </h3>
        <p className="text-center text-sm max-w-md text-gray-500 dark:text-gray-400 leading-relaxed">
          Upload documents to begin intelligent document analysis, or ask any question to explore research topics
        </p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="py-6 px-4 space-y-1 min-h-full">
      {displayMessages.map((msg) => (
        <Message key={msg.id} message={msg} />
      ))}
      <div ref={messagesEndRef} className="h-4" />
    </div>
  );
};

export default MessageList;
