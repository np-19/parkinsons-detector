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
      className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition-all ${
        isDragOver
          ? "border-blue-500 bg-blue-50/50"
          : "border-slate-200 hover:border-slate-300 hover:bg-slate-50/50"
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

      <div className="w-10 h-10 rounded-full bg-slate-100 text-slate-500 flex items-center justify-center mb-2">
        <UploadCloud className="w-5 h-5 text-blue-600" />
      </div>

      <div className="text-sm font-medium text-slate-800 text-center">
        Drop audio recording(s) here, or <span className="text-blue-600 underline">browse files</span>
      </div>
      <div className="text-xs text-slate-400 mt-1 flex items-center gap-1.5">
        <FileAudio className="w-3.5 h-3.5" />
        Supports .wav, .m4a, .ogg, .mpeg, .webm (Up to 25 MB)
      </div>
    </div>
  );
};
