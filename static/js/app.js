$(document).ready(function () {
  // --------- Estado ---------
  let currentFile = null;
  let uploadedFilePath = null;
  let proposals = [];
  let currentSessionId = null;

  // --------- Elementos ---------
  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");
  const btnSelect = $("#btnSelect");
  const btnAnalyze = $("#btnAnalyze");
  const btnRender = $("#btnRender");
  const btnZip = $("#btnZip");
  const uploadInfo = $("#uploadInfo");
  const progUpload = $("#progUpload");
  const proposalsContainer = $("#proposals");
  const renderProgressWrap = $("#renderProgressWrap");
  const renderProgress = $("#renderProgress");
  const renderMsg = $("#renderMsg");

  // Modal (editor)
  const editorModal = new bootstrap.Modal("#editorModal");
  const editorVideo = $("#editorVideo")[0];
  const timeSlider = $("#timeSlider");
  const curTimeLabel = $("#curTimeLabel");
  const kfX = $("#kfX");
  const kfY = $("#kfY");
  const kfT = $("#kfT");
  const btnAddKf = $("#btnAddKf");
  const kfList = $("#kfList");
  const clipStart = $("#clipStart");
  const clipEnd = $("#clipEnd");
  const btnSaveEdit = $("#btnSaveEdit");
  const modalClipTitle = $("#modalClipTitle");

  // Timeline custom
  const timelineWrapper = $("#timelineWrapper");
  const handleStart = $("#handleStart");
  const handleEnd = $("#handleEnd");
  const selection = $("#selection");
  const rangeStartLabel = $("#rangeStartLabel");
  const rangeEndLabel = $("#rangeEndLabel");
  const btnPreviewCut = $("#btnPreviewCut");

  const btnYoutube = $("#btnYoutube");
  const youtubeUrl = $("#youtubeUrl");

  let currentEditingClip = null;
  let currentKeyframes = [];
  let videoDuration = 0;
  let draggingHandle = null;

  // --------- Upload ---------
  btnSelect.on("click", () => fileInput.click());
  dropzone.on("click", () => fileInput.click());
  dropzone.on("dragover", (e) => {
    e.preventDefault();
    dropzone.addClass("dropzone-active");
  });
  dropzone.on("dragleave", (e) => {
    e.preventDefault();
    dropzone.removeClass("dropzone-active");
  });
  dropzone.on("drop", (e) => {
    e.preventDefault();
    dropzone.removeClass("dropzone-active");
    if (e.originalEvent.dataTransfer.files.length > 0) {
      handleFileUpload(e.originalEvent.dataTransfer.files[0]);
    }
  });
  fileInput.on("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  //---------- Upload YouTube ---------
btnYoutube.on("click", function () {
  const url = youtubeUrl.val().trim();
  if (!url) {
    alert("Cole um link válido do YouTube.");
    return;
  }

  uploadInfo.text("Iniciando download do YouTube...");
  btnAnalyze.prop("disabled", true);
  btnRender.prop("disabled", true);
  btnZip.prop("disabled", true);
  proposalsContainer.empty();

  $.ajax({
    url: "/upload_youtube",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({ url }),
    success: function (response) {
      const sessionId = response.session_id;
      const evtSource = new EventSource(`/youtube_status/${sessionId}`);

      evtSource.onmessage = function (event) {
        const data = JSON.parse(event.data);
        uploadInfo.text(data.message);

        if (data.status === "done") {
          // 🔧 corrigido: usar o caminho completo em vez de só o nome
          uploadedFilePath = data.filepath;  
          console.log("Download concluído:", uploadedFilePath);
          currentFile = uploadedFilePath;

          btnAnalyze.prop("disabled", false);
          evtSource.close();
        }

        if (data.status === "error") {
          alert("Erro ao baixar: " + data.message);
          evtSource.close();
        }
      };
    },
    error: function (xhr) {
      alert("Erro ao iniciar download do YouTube: " + (xhr.responseJSON?.error || "desconhecido"));
    }
  });
});



//------------Keyframes ---------
function updateCropOverlay(time = editorVideo.currentTime) {
  if (!editorVideo.videoWidth || !editorVideo.videoHeight) return;

  const videoW = editorVideo.clientWidth;
  const videoH = editorVideo.clientHeight;
  const cropH = videoH; // sempre altura cheia
  const cropW = cropH * 9 / 16;

  // 🔹 interpolação entre keyframes
  let x = 0.5, y = 0.5;
  if (currentKeyframes.length > 0) {
    const before = [...currentKeyframes].reverse().find(kf => kf.time <= time);
    const after = currentKeyframes.find(kf => kf.time >= time);

    if (before && after && before !== after) {
      const ratio = (time - before.time) / (after.time - before.time);
      x = before.x + ratio * (after.x - before.x);
      y = before.y + ratio * (after.y - before.y);
    } else if (before) {
      x = before.x; y = before.y;
    } else if (after) {
      x = after.x; y = after.y;
    }
  }

  // converter para pixels
  const left = (x * videoW) - cropW / 2;
  const top = (y * videoH) - cropH / 2;

  $("#cropOverlay").css({
    left: `${Math.max(0, Math.min(videoW - cropW, left))}px`,
    top: `${Math.max(0, Math.min(videoH - cropH, top))}px`,
    width: `${cropW}px`,
    height: `${cropH}px`
  });
}

editorVideo.addEventListener("timeupdate", () => updateCropOverlay());
timeSlider.on("input", () => updateCropOverlay(parseFloat(timeSlider.val())));

btnAddKf.on("click", () => {
  addKeyframe();
  updateCropOverlay();
});



//--------- Fim Keyframes ---------
  

  function handleFileUpload(file) {
    if (!file.type.match("video.*")) {
      alert("Por favor, selecione um arquivo de vídeo válido.");
      return;
    }
    currentFile = file;
    uploadInfo.text(`${file.name} (${formatFileSize(file.size)})`);
    btnAnalyze.prop("disabled", true);
    btnRender.prop("disabled", true);
    btnZip.prop("disabled", true);
    proposalsContainer.empty();
    uploadedFilePath = null;

    const formData = new FormData();
    formData.append("file", file);

    progUpload.removeClass("d-none");
    const progressBar = progUpload.find(".progress-bar");
    progressBar.css("width", "0%");

    $.ajax({
      url: "/upload",
      type: "POST",
      data: formData,
      processData: false,
      contentType: false,
      xhr: function () {
        const xhr = new window.XMLHttpRequest();
        xhr.upload.addEventListener(
          "progress",
          function (evt) {
            if (evt.lengthComputable) {
              const percentComplete = (evt.loaded / evt.total) * 100;
              progressBar.css("width", percentComplete + "%");
            }
          },
          false
        );
        return xhr;
      },
      success: function (response) {
        uploadedFilePath = response.filepath;
        uploadInfo.text(`${response.filename} carregado com sucesso.`);
        btnAnalyze.prop("disabled", false);
        progUpload.addClass("d-none");
        progressBar.css("width", "0%");
      },
      error: function () {
        alert("Erro ao fazer upload do arquivo.");
        btnAnalyze.prop("disabled", true);
        progUpload.addClass("d-none");
        progressBar.css("width", "0%");
      },
    });
  }

  function formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  // --------- Análise ---------
  btnAnalyze.on("click", analyzeVideo);

  function analyzeVideo() {
    if (!uploadedFilePath) {
      alert("Faça o upload do arquivo antes de analisar.");
      return;
    }

    console.log("Analisando:", uploadedFilePath);

    btnAnalyze.prop("disabled", true);
    btnAnalyze.html(
      '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analisando...'
    );

    $.ajax({
      url: "/analyze",
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify({ filepath: uploadedFilePath }),
      success: function (response) {
        const sessionId = response.session_id;
        const evtSource = new EventSource(`/analyze_status/${sessionId}`);

        evtSource.onmessage = function (event) {
          const data = JSON.parse(event.data);
          // console.log("Analyze progress:", data);

          // Atualiza mensagem
          uploadInfo.text(data.message || "");

          // Atualiza progresso (se quiser usar barra de renderização existente)
          if (data.total !== undefined) {
            renderProgressWrap.removeClass("d-none");
            renderProgress.css("width", `${data.total}%`);
            // renderProgress.text(`${data.total}%`);
          }

          if (data.status === "done") {
            proposals = data.proposals || [];
            console.log("Proposals:", proposals);
            displayProposals();
            btnAnalyze.prop("disabled", false).text("Gerar cortes");
            btnRender.prop("disabled", proposals.length === 0);
            btnZip.prop("disabled", true);
            evtSource.close();
          }

          if (data.status === "error") {
            alert("Erro na análise: " + data.message);
            btnAnalyze.prop("disabled", false).text("Gerar cortes");
            evtSource.close();
          }
        };
      },
      error: function () {
        alert("Erro ao iniciar análise.");
        btnAnalyze.prop("disabled", false);
        btnAnalyze.text("Gerar cortes");
      },
    });
  }


  function displayProposals() {
    proposalsContainer.empty();

    if (proposals.length === 0) {
      proposalsContainer.append(
        '<div class="list-group-item text-secondary">Nenhum corte interessante encontrado.</div>'
      );
      return;
    }

    proposals.forEach((proposal, index) => {
      const startTime = formatTime(proposal.start);
      const endTime = formatTime(proposal.end);
      const duration = formatTime(proposal.end - proposal.start);

      const item = $(`
        <div class="list-group-item" data-clip-id="${proposal.id}">
          <div class="d-flex justify-content-between align-items-center">
            <div>
              <input type="checkbox" class="clip-select me-2" checked>
              <strong>Corte #${index + 1}</strong>
              <div class="text-secondary small">${startTime} → ${endTime} (${duration})</div>
            </div>
            <div class="d-flex gap-2 align-items-center">
              <span class="badge text-bg-secondary small clip-badge" data-clip-badge="${proposal.id}">0%</span>
              <button class="btn btn-sm btn-outline-primary btn-edit">Editar</button>
            </div>
          </div>
          <input type="hidden" class="clip-start" value="${proposal.start}">
          <input type="hidden" class="clip-end" value="${proposal.end}">
        </div>
      `);

      proposalsContainer.append(item);
    });
  }

  function formatTime(seconds) {
    if (seconds < 0) seconds = 0;
    const date = new Date(seconds * 1000);
    if (seconds >= 3600) {
      return date.toISOString().substr(11, 8);
    }
    return date.toISOString().substr(14, 5);
  }

  // --------- Editor ---------
  proposalsContainer.off("click", ".btn-edit").on("click", ".btn-edit", function () {
    const clipId = $(this).closest(".list-group-item").data("clip-id");
    console.log("Editando corte:", clipId);
    console.log("proposals:", proposals);
    const proposal = proposals.find((p) => p.id === clipId);
    if (proposal) openEditor(proposal);
  });

  function openEditor(clip) {
    currentEditingClip = clip;
    currentKeyframes = (clip.keyframes || []).map(kf => ({
      time: kf.time !== undefined ? kf.time : kf.t, // aceita time ou t
      x: kf.x,
      y: kf.y
    }));

    kfList.empty();

    console.log("Editando corte:", clip);
    console.log("currentFile:", currentFile);

    modalClipTitle.text(`#${proposals.findIndex((p) => p.id === clip.id) + 1}`);
    // editorVideo.src = URL.createObjectURL(currentFile);
    if (currentFile instanceof File) {
      // Upload local → cria URL temporária
      editorVideo.src = URL.createObjectURL(currentFile);
    } else if (uploadedFilePath) {
      // Arquivo do servidor (YouTube ou upload processado) → monta URL pública
      const filename = uploadedFilePath.split("/").pop();
      editorVideo.src = `/uploads/${filename}`;
    } else {
      alert("Nenhum vídeo carregado para edição.");
      return;
    }

    editorVideo.onloadedmetadata = function () {
      videoDuration = editorVideo.duration;

      // inicia handles no corte atual
      const startPercent = (clip.start / videoDuration) * 100;
      const endPercent = (clip.end / videoDuration) * 100;

      handleStart.css("left", startPercent + "%");
      handleEnd.css("left", endPercent + "%");
      updateSelection();

      clipStart.val(clip.start.toFixed(1));
      clipEnd.val(clip.end.toFixed(1));

      timeSlider.attr("max", videoDuration.toFixed(1));
      editorVideo.currentTime = clip.start;
      timeSlider.val(clip.start.toFixed(1));
      updateTimeLabel();
    };

    updateKeyframesList();
    editorModal.show();
  }

  function updateTimeLabel() {
    const rel = parseFloat(timeSlider.val() || "0");
    kfT.val(rel.toFixed(1));
  }

  timeSlider.on("input", function () {
    if (editorVideo && currentEditingClip) {
      const rel = parseFloat(timeSlider.val());
      editorVideo.currentTime = rel;
      updateTimeLabel();
    }
  });

  editorVideo.addEventListener("timeupdate", function () {
    if (!timeSlider.is(":focus")) {
      timeSlider.val(editorVideo.currentTime.toFixed(1));
      updateTimeLabel();
    }
  });

  function secondsToHMS(totalSeconds) {
    totalSeconds = Math.floor(totalSeconds); // arredonda pra baixo
    const hours   = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    // formata sempre com 2 dígitos
    const hh = hours.toString().padStart(2, '0');
    const mm = minutes.toString().padStart(2, '0');
    const ss = seconds.toString().padStart(2, '0');

    return `${hh}:${mm}:${ss}`;
}


  function updateSelection() {
    const startPx = handleStart.position().left;
    const endPx = handleEnd.position().left;
    const barWidth = timelineWrapper.width();

    const startTime = (startPx / barWidth) * videoDuration;
    const endTime = (endPx / barWidth) * videoDuration;

    selection.css({
      left: startPx + "px",
      width: endPx - startPx + "px",
    });

    clipStart.val(startTime.toFixed(1));
    clipEnd.val(endTime.toFixed(1));

    rangeStartLabel.text(secondsToHMS(startTime));
    rangeEndLabel.text(secondsToHMS(endTime));

    curTimeLabel.text((endTime-startTime).toFixed(1) + "s");
  }

  // Drag handles
  $(".handle").on("mousedown", function (e) {
    draggingHandle = $(this);
    e.preventDefault();
  });

  $(document).on("mousemove", function (e) {
    if (!draggingHandle) return;

    const wrapperOffset = timelineWrapper.offset().left;
    const wrapperWidth = timelineWrapper.width();
    let newLeft = e.pageX - wrapperOffset;

    if (draggingHandle.is(handleStart)) {
      const maxLeft = handleEnd.position().left - 10;
      newLeft = Math.max(0, Math.min(newLeft, maxLeft));
    } else {
      const minLeft = handleStart.position().left + 10;
      newLeft = Math.max(minLeft, Math.min(newLeft, wrapperWidth));
    }

    draggingHandle.css("left", newLeft + "px");
    updateSelection();
  });

  $(document).on("mouseup", function () {
    draggingHandle = null;
  });

  // Preview corte
  btnPreviewCut.on("click", function () {
    const start = parseFloat(clipStart.val());
    const end = parseFloat(clipEnd.val());
    editorVideo.currentTime = start;
    editorVideo.play();

    const stopPlayback = () => {
      if (editorVideo.currentTime >= end) {
        editorVideo.pause();
        editorVideo.removeEventListener("timeupdate", stopPlayback);
      }
    };
    editorVideo.addEventListener("timeupdate", stopPlayback);
  });

  function addKeyframe() {
    const x = parseFloat(kfX.val());
    const y = parseFloat(kfY.val());
    const t = parseFloat(kfT.val());
    if ([x, y, t].some((v) => Number.isNaN(v))) {
      alert("Valores inválidos para keyframe");
      return;
    }
    if (x < 0 || x > 1 || y < 0 || y > 1) {
      alert("X e Y devem estar entre 0 e 1");
      return;
    }
    currentKeyframes.push({ x, y, time: t });
    updateKeyframesList();
  }

  function updateKeyframesList() {
      kfList.empty();

      currentKeyframes
        .sort((a, b) => a.time - b.time)
        .forEach((kf, index) => {
          const item = $(`
            <li class="list-group-item d-flex justify-content-between align-items-center py-2 keyframe-item" style="cursor:pointer;">
              <div class="d-flex gap-2 align-items-center small flex-grow-1">
                <span class="badge bg-primary">${index + 1}</span>
                T: <input type="number" step="0.1" class="form-control form-control-sm kf-edit-time disabled" style="width:70px" value="${kf.time.toFixed(1)}">
                X: <input type="number" step="0.01" min="0" max="1" class="form-control form-control-sm kf-edit-x" style="width:70px" value="${kf.x.toFixed(2)}">
                Y: <input type="number" step="0.01" min="0" max="1" class="form-control form-control-sm kf-edit-y" style="width:70px" value="${kf.y.toFixed(2)}">
              </div>
              <button class="btn btn-sm btn-outline-danger btn-remove-kf" data-index="${index}">&times;</button>
            </li>
          `);

          // 🔹 clicar no item → pular para o tempo do keyframe
          item.on("click", function (e) {
            if ($(e.target).is("input") || $(e.target).is("button")) return;

            // 🔹 Remove destaque de todos
            $(".keyframe-item").removeClass("active");

            // 🔹 Destaca este
            $(this).addClass("active");

            // Pula para o tempo do keyframe
            editorVideo.currentTime = kf.time;
            timeSlider.val(kf.time.toFixed(1));
            updateCropOverlay(kf.time);
          });

          // 🔹 listeners para edição
          item.find(".kf-edit-time").on("change", function () {
            currentKeyframes[index].time = parseFloat($(this).val()) || 0;
            updateCropOverlay(currentKeyframes[index].time);
          });

          item.find(".kf-edit-x").on("change", function () {
            currentKeyframes[index].x = Math.min(1, Math.max(0, parseFloat($(this).val()) || 0.5));
            updateCropOverlay(currentKeyframes[index].time);
          });
          
          item.find(".kf-edit-y").on("change", function () {
            currentKeyframes[index].y = Math.min(1, Math.max(0, parseFloat($(this).val()) || 0.5));
            updateCropOverlay(currentKeyframes[index].time);
          });

          // 🔹 remover
          item.find(".btn-remove-kf").on("click", function () {
            currentKeyframes.splice(index, 1);
            updateKeyframesList();
            updateCropOverlay();
          });

          kfList.append(item);
        });
    }



  btnSaveEdit.on("click", saveClipEdits);
  function saveClipEdits() {
    if (!currentEditingClip) return;

    const start = parseFloat(clipStart.val());
    const end = parseFloat(clipEnd.val());
    if (Number.isNaN(start) || Number.isNaN(end) || start >= end) {
      alert("Valores de início/fim inválidos");
      return;
    }

    const proposalIndex = proposals.findIndex((p) => p.id === currentEditingClip.id);
    if (proposalIndex !== -1) {
      proposals[proposalIndex].start = start;
      proposals[proposalIndex].end = end;
      proposals[proposalIndex].keyframes = currentKeyframes.map(kf => ({
        t: kf.time,
        x: kf.x,
        y: kf.y
      }));

      const item = proposalsContainer.find(`[data-clip-id="${currentEditingClip.id}"]`);
      item.find(".clip-start").val(start);
      item.find(".clip-end").val(end);

      const startTime = formatTime(start);
      const endTime = formatTime(end);
      const duration = formatTime(end - start);

      item.find("strong").text(`Corte #${proposalIndex + 1}${currentKeyframes.length > 0 ? " (editado)" : ""}`);
      item.find(".small").first().text(`${startTime} → ${endTime} (${duration})`);
    }

    editorModal.hide();
  }

  // --------- Render (progresso real via SSE) ---------
  btnRender.on("click", renderClips);

  function renderClips() {
    if (proposals.length === 0 || !uploadedFilePath) return;

    btnRender.prop("disabled", true);
    renderProgressWrap.removeClass("d-none");
    renderProgress.css("width", "0%");
    renderProgress.text("");
    renderMsg.text("Preparando para renderizar...");

    $.ajax({
      url: "/render",
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify({
        filepath: uploadedFilePath,
        clips: proposals
          .filter((p) => {
            const item = proposalsContainer.find(`[data-clip-id="${p.id}"]`);
            return item.find(".clip-select").is(":checked"); // só os checados
          })
          .map((p) => ({
            id: p.id,
            start: p.start,
            end: p.end,
            keyframes: p.keyframes || [],
          })),
      }),
      success: function (response) {
        currentSessionId = response.session_id;
        listenRenderProgress(currentSessionId);
      },
      error: function () {
        alert("Erro ao iniciar renderização.");
        btnRender.prop("disabled", false);
        renderMsg.html('<span class="text-danger">Erro na renderização</span>');
      },
    });
  }

  function listenRenderProgress(sessionId) {
    const evt = new EventSource(`/render_status/${sessionId}`);
    evt.onmessage = function (event) {
      try {
        const data = JSON.parse(event.data);
        const total = Number(data.total || 0);
        const status = data.status || "running";
        const message = data.message || "";

        renderProgress.css("width", `${total}%`);
        renderProgress.text(`${total}%`);
        renderMsg.text(message);

        if (data.clips) {
          Object.entries(data.clips).forEach(([clipId, info]) => {
            const pct = info.percent ?? 0;
            const badge = $(`[data-clip-badge="${clipId}"]`);
            if (badge.length) badge.text(`${pct}%`);
          });
        }

        if (status === "done") {
          evt.close();
          renderMsg.html('<span class="text-success">Renderização concluída!</span>');
          btnZip.prop("disabled", false);

          proposalsContainer.find(".list-group-item").each(function () {
            const clipId = $(this).data("clip-id");
            if ($(this).find(".btn-outline-success").length === 0) {
              const clipUrl = `/download_clip/${sessionId}/${clipId}`;
              $(this)
                .find(".d-flex")
                .append(`<a href="${clipUrl}" class="btn btn-sm btn-outline-success">Download</a>`);
              $(this).append(`
                <div class="mt-2">
                  <video src="${clipUrl}" controls preload="metadata" style="max-width:100%; height:auto;"></video>
                </div>
              `);
            }
          });
        } else if (status === "error") {
          evt.close();
          renderMsg.html(`<span class="text-danger">${message}</span>`);
          btnRender.prop("disabled", false);
        }
      } catch (e) {
        console.error("SSE parse error:", e);
      }
    };
  }

    // --------- Drag do cropOverlay ---------
  let draggingOverlay = false;
  let activeKeyframeIndex = null;

  // quando clica num keyframe da lista → seleciona
  kfList.on("click", ".keyframe-item", function (e) {
    if ($(e.target).is("input") || $(e.target).is("button")) return;

    $(".keyframe-item").removeClass("active");
    $(this).addClass("active");
    activeKeyframeIndex = $(this).index();

    const kf = currentKeyframes[activeKeyframeIndex];
    editorVideo.currentTime = kf.time;
    timeSlider.val(kf.time.toFixed(1));
    updateCropOverlay(kf.time);
  });

  // início do drag
  $("#cropOverlay").on("mousedown", function (e) {
    if (activeKeyframeIndex === null) return; // precisa ter keyframe selecionado
    draggingOverlay = true;
    e.preventDefault();
  });

  // movimento
  $(document).on("mousemove", function (e) {
    if (!draggingOverlay || activeKeyframeIndex === null) return;

    const rect = $("#editorVideo").offset();
    const videoW = editorVideo.clientWidth;
    const videoH = editorVideo.clientHeight;

    // posição do mouse relativa ao vídeo
    let relX = (e.pageX - rect.left) / videoW;
    let relY = (e.pageY - rect.top) / videoH;

    // clampa entre 0 e 1
    relX = Math.min(1, Math.max(0, relX));
    relY = Math.min(1, Math.max(0, relY));

    // atualiza keyframe ativo
    currentKeyframes[activeKeyframeIndex].x = relX;
    currentKeyframes[activeKeyframeIndex].y = relY;

    // atualiza overlay e inputs
    updateCropOverlay(editorVideo.currentTime);
    updateKeyframesList();

    // mantém item selecionado com highlight
    kfList.find(".keyframe-item").removeClass("active")
      .eq(activeKeyframeIndex).addClass("active");
  });

  // fim do drag
  $(document).on("mouseup", function () {
    draggingOverlay = false;
  });


  btnZip.on("click", function () {
    if (currentSessionId) {
      window.location.href = `/download/${currentSessionId}`;
    }
  });
});
