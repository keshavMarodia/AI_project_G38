import { React, useState } from "react";
import "./Main.css";

const Main = (props) => {
  const [probab,setProbab]= useState("");
  const [text, setText] = useState("");
  const [astar, setAstar] = useState("");
  const [bfs, setBfs] = useState("");
  const [dfs, setDfs] = useState("");
  const [ga,setGa]=useState("");

  const dataCapture = () => {
    fetch("http://127.0.0.1:5000/fetch_values")
      .then((response) => response.json())
      .then((data) => {
        console.log(data)
        setProbab(data["values"][1]);
        setText(data["values"][0]);
        setAstar(data["value_astar"]);
        setBfs(data["value_bfs"]);
        setDfs(data["value_dfs"]);
        setGa(data["value_ga"]);
      })
      .catch((error) => console.log(error));
    return;
  };

  return (
    <div className="container">
      <nav>
        <ul>
          <div className="left">
            <li><a href="#">Home</a></li>
            <li><a href="#">About us</a></li>
            <li><a href="#">Scan Prescription</a></li>
            <li><a href="#">Algorithms used</a></li>
          </div>
          <li className="logout"><a href="#">Logout</a></li>
        </ul>
      </nav>
      <div className="content">
        <label className="label">Text : {text}</label>
        <label className="label">Probability : {probab}</label>
        <label className="label">text astar : {astar}</label>
        <label className="label">text bfs : {bfs}</label>
        <label className="label">text dfs : {dfs}</label>
        <label className="label">text GA : {ga}</label>
        <button className="button" onClick={dataCapture}>Fetch Data</button>
      </div>
    </div>
  );
};

export default Main;