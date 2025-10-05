import React, { useEffect, useRef } from 'react';
import Message from './Message';
import { useChatContext } from '../context/ChatContext';

const MessageList = () => {
  const { messages } = useChatContext();
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Filter out system messages for display
  const displayMessages = messages.filter(msg => msg.role !== 'system');

  if (displayMessages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500 bg-white">
        <div className="text-center">
          <p className="text-lg mb-2">No messages yet</p>
          <p className="text-sm">Upload documents or start a conversation</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 bg-white">
      {displayMessages.map(message => (
        <Message key={message.id} message={message} />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessageList;