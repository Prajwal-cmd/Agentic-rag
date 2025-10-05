import React, { useState, useRef } from 'react';
import { Upload, X, File, AlertCircle, CheckCircle } from 'lucide-react';
import { useChatContext } from '../context/ChatContext';
import { chatAPI } from '../api/apiClient';
import { formatFileSize, isValidFileType } from '../utils/formatters';

const DocumentUpload = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const fileInputRef = useRef(null);
  
  const { sessionId, setDocumentsUploaded, addMessage } = useChatContext();
  const maxSize = 15 * 1024 * 1024; // 15MB

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    handleFiles(files);
  };

  const handleFiles = (files) => {
    setUploadError(null);
    setUploadSuccess(null);

    // Validate file types
    const invalidFiles = files.filter((f) => !isValidFileType(f.name));
    if (invalidFiles.length > 0) {
      setUploadError(`Invalid file types: ${invalidFiles.map((f) => f.name).join(', ')}`);
      return;
    }

    // Validate total size
    const totalSize = files.reduce((sum, f) => sum + f.size, 0);
    if (totalSize > maxSize) {
      setUploadError(`Total file size exceeds 15MB limit (${formatFileSize(totalSize)})`);
      return;
    }

    setSelectedFiles(files);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const removeFile = (index) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setUploadError(null);
    setUploadSuccess(null);
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0 || !sessionId) return;

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      console.log('Uploading with session ID:', sessionId);
      const result = await chatAPI.uploadDocuments(selectedFiles, sessionId);
      console.log('Upload result:', result);

      // Verify session IDs match
      if (result.session_id !== sessionId) {
        console.error(`Session ID mismatch: sent ${sessionId}, got ${result.session_id}`);
        setUploadError('Session ID mismatch error. Please refresh the page.');
        return;
      }

      const successMsg = `Successfully uploaded ${result.files_processed} file(s), created ${result.chunks_created} chunks`;
      setUploadSuccess(successMsg);
      addMessage('system', `✓ ${successMsg}`);
      setDocumentsUploaded(true);
      setSelectedFiles([]);
      
      // Clear success message after 5 seconds
      setTimeout(() => setUploadSuccess(null), 5000);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message || 'Upload failed';
      setUploadError(errorMsg);
      console.error('Upload error:', error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="p-4">
      <h2 className="text-lg font-semibold mb-4 text-gray-900">Upload Documents</h2>

      {/* Drop zone */}
      <div
        className={`border-2 border-dashed rounded-lg p-6 mb-4 text-center cursor-pointer transition-colors ${
          uploading 
            ? 'border-gray-200 bg-gray-50 cursor-not-allowed' 
            : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
        }`}
        onClick={() => !uploading && fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <Upload className={`mx-auto mb-2 ${uploading ? 'text-gray-300' : 'text-gray-400'}`} size={32} />
        <p className="text-sm text-gray-600 mb-1 font-medium">
          {uploading ? 'Uploading...' : 'Click or drag files here'}
        </p>
        <p className="text-xs text-gray-500">
          PDF, DOCX, TXT (max 15MB total)
        </p>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt"
          onChange={handleFileSelect}
          className="hidden"
          disabled={uploading}
        />
      </div>

      {/* Selected files list */}
      {selectedFiles.length > 0 && (
        <div className="mb-4">
          <p className="text-sm font-medium mb-2 text-gray-700">
            Selected files ({selectedFiles.length}):
          </p>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {selectedFiles.map((file, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-2 bg-gray-50 rounded border border-gray-200"
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <File size={16} className="text-gray-500 flex-shrink-0" />
                  <span className="text-sm truncate text-gray-700">{file.name}</span>
                  <span className="text-xs text-gray-500 flex-shrink-0">
                    {formatFileSize(file.size)}
                  </span>
                </div>
                <button
                  onClick={() => removeFile(idx)}
                  disabled={uploading}
                  className="text-red-500 hover:text-red-700 flex-shrink-0 ml-2 disabled:opacity-50"
                >
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>

          <button
            onClick={handleUpload}
            disabled={uploading || selectedFiles.length === 0}
            className="w-full mt-3 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
          >
            {uploading ? (
              <span className="flex items-center justify-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Uploading...
              </span>
            ) : (
              'Upload Files'
            )}
          </button>
        </div>
      )}

      {/* Success message */}
      {uploadSuccess && (
        <div className="flex items-start gap-2 p-3 bg-green-50 border border-green-200 rounded-lg text-sm text-green-700 mb-4">
          <CheckCircle size={16} className="flex-shrink-0 mt-0.5" />
          <p>{uploadSuccess}</p>
        </div>
      )}

      {/* Error display */}
      {uploadError && (
        <div className="flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700 mb-4">
          <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
          <p>{uploadError}</p>
        </div>
      )}

      {/* Info */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-600 mb-1">
          <strong>Supported formats:</strong> PDF, DOCX, TXT
        </p>
        <p className="text-xs text-gray-600">
          <strong>Max size:</strong> 15MB total
        </p>
      </div>
    </div>
  );
};

export default DocumentUpload;