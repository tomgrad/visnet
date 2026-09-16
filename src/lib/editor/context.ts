export interface EditorNodeActions {
  removeBlock: (id: string) => void;
}

export const EDITOR_NODE_ACTIONS = Symbol('visnet-editor-node-actions');
