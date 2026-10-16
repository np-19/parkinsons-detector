import React, { useRef, useState } from "react";
import { UploadCloud, FileAudio } from "lucide-react";

interface DropzoneProps {
  onFilesSelected: (files: File[]) => void;
}

export const Dropzone: React.FC<DropzoneProps> = ({ onFilesSelected }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const validFiles = Array.from(e.dataTransfer.files).filter((file) =>
        /\.(wav|mp3|mpeg|mpg|ogg|m4a|aac|flac|webm)$/i.test(file.name)
      );
      if (validFiles.length > 0) {
        onFilesSelected(validFiles);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(Array.from(e.target.files));
    }
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
      className={`border-2 border-dashed rounded-2xl py-7 px-4 sm:p-8 flex flex-col items-center justify-center cursor-pointer transition-all touch-manipulation select-none ${
        isDragOver
          ? "border-blue-500 bg-blue-50/70 scale-[0.99]"
          : "border-slate-200/90 hover:border-blue-300 hover:bg-slate-50/60 active:bg-slate-100"
      }`}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".wav,.mp3,.mpeg,.mpg,.ogg,.m4a,.aac,.flac,.webm"
        onChange={handleFileChange}
        className="hidden"
      />

      <div className="w-12 h-12 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center mb-3 border border-blue-100 shadow-2xs">
        <UploadCloud className="w-6 h-6 text-blue-600" />
      </div>

      <div className="text-sm font-semibold text-slate-800 text-center">
        Drop audio recordings here, or <span className="text-blue-600 underline underline-offset-2">browse files</span>
      </div>
      <div className="text-xs text-slate-400 mt-1.5 flex items-center gap-1.5 text-center">
        <FileAudio className="w-3.5 h-3.5 text-slate-400 shrink-0" />
        <span>Supports WAV, M4A, OGG, MP3, FLAC, WEBM (Up to 25 MB)</span>
      </div>
    </div>
  );
};

