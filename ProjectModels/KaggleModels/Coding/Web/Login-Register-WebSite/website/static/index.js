function deleteNote(noteId,fileId) {
  fetch("/delete-note", {
    method: "POST",
    body: JSON.stringify({ noteId: noteId , fileId: fileId}),
  }).then((_res) => {
    window.location.href = "/";
  });
}