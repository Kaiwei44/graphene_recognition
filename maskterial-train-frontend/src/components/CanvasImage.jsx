import { useEffect, useRef } from "react";
import styles from "./CanvasImage.module.css"; // Replace with your actual styles file

export function CanvasImage({ src, flakes, ...rest }) {
  const canvasRef = useRef(null);

  const colorMap = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
  ];

  useEffect(() => {
    if (!src) return;

    const img = new Image();

    img.onload = () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      const parent = canvas.parentElement;

      console.log(img.width, img.height);

      let img_aspectRatio = img.width / img.height;
      let canvas_aspectRatio = parent.offsetWidth / parent.offsetHeight;

      if (img_aspectRatio > canvas_aspectRatio) {
        canvas.width = parent.offsetWidth;
        canvas.height = canvas.width / img_aspectRatio;
      } else {
        canvas.height = parent.offsetHeight;
        canvas.width = canvas.height * img_aspectRatio;
      }

      // Calculate aspect ratio to scale the image proportionally
      const scale = Math.min(
        canvas.width / img.width,
        canvas.height / img.height
      );
      const x = (canvas.width - img.width * scale) / 2;
      const y = (canvas.height - img.height * scale) / 2;

      ctx.drawImage(img, x, y, img.width * scale, img.height * scale); // Draw image

      flakes.forEach((flake, index) => {
        console.log(flake, index, colorMap[index % colorMap.length]);
        ctx.strokeStyle = colorMap[index % colorMap.length];
        ctx.fillStyle = colorMap[index % colorMap.length];
        ctx.lineWidth = 2;

        // draw bounding box
        let [x1, y1, dx, dy] = flake.bbox;
        ctx.beginPath();
        ctx.roundRect(
          x1 * scale + x,
          y1 * scale + y,
          dx * scale,
          dy * scale,
          5
        );
        ctx.stroke();
      });
    };

    img.src = src; // Set the image source
  }, [src, flakes]);

  const flakeTable = (
    <div className={styles.flakeDisplay}>
      <table className={styles.flakeTable}>
        <thead>
          <tr>
            <th>Color</th>
            <th>Class</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>
          {flakes.map((flake, index) => (
            <tr key={index}>
              <td>
                <div
                  className={styles.colorBox}
                  style={{
                    backgroundColor: colorMap[index % colorMap.length],
                  }}
                ></div>
              </td>
              <td>{flake.thickness}</td>
              <td>
                {Math.round(
                  (1 - flake.false_positive_probability).toFixed(3) * 100
                )}
                %
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return (
    <>
      <canvas ref={canvasRef} className={styles.canvas}></canvas>
      {flakes.length > 0 && flakeTable}
    </>
  );
}
